"""Exact split reuse and streaming 224-frame video loading."""

from collections import OrderedDict
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex

from .config import FRAMES_PER_CLIP, TARGET


REQUIRED_COLUMNS = {
    "hospital_id", "video_id", "split", "raw_value", "binary_label",
    "score_threshold", "mirror", "lab_patient_id",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_records(reference_output_dir, output_dir):
    source = os.path.join(reference_output_dir, "task_records", f"{TARGET}.csv")
    records = pd.read_csv(source, dtype={"hospital_id": str, "video_id": str})
    missing = sorted(REQUIRED_COLUMNS - set(records.columns))
    if missing:
        raise ValueError(f"Reference records lack columns: {missing}")
    if records["video_id"].duplicated().any():
        raise ValueError("Reference records contain duplicate videos")
    if records.groupby("hospital_id")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage exists in the reference split")
    if set(records["split"]) != {"train", "val", "test"}:
        raise ValueError("Reference split must contain train, val, and test")
    records = records.copy()
    records["hemoglobin_g_dl"] = pd.to_numeric(records["raw_value"], errors="raise") / 10.0
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, "task_records.csv")
    records.to_csv(destination, index=False)
    return records, source, destination


def validate_index(records, index_dir):
    index_path = os.path.join(index_dir, "frame_offsets.npz")
    manifest_path = os.path.join(index_dir, "index_manifest.json")
    if not os.path.isfile(index_path) or not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "The reusable all-frame byte-offset index is missing; build the existing "
            "face regression all-frame index first"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("frame_policy", {}).get("mode") != "all_decodable_frames":
        raise RuntimeError("Configured index is not an all-frame index")
    index = FrameOffsetIndex.load(index_path)
    missing = sorted(set(records["video_id"]) - set(index.video_lookup))
    if missing:
        raise RuntimeError(f"All-frame index lacks {len(missing)} reference videos")
    return index, index_path, manifest_path


def selected_global_indices(index, video_id):
    start, end = index.frame_range(video_id)
    count = end - start
    if count < 1:
        raise ValueError(f"No indexed frames for {video_id}")
    local = np.rint(np.linspace(0, count - 1, FRAMES_PER_CLIP)).astype(np.int64)
    return start + local, count, int(FRAMES_PER_CLIP - len(np.unique(local)))


def write_sampling_audit(records, index, output_dir):
    rows = []
    for row in records.itertuples(index=False):
        selected, source_count, repeats = selected_global_indices(index, row.video_id)
        rows.append({
            "hospital_id": row.hospital_id,
            "video_id": row.video_id,
            "split": row.split,
            "source_decodable_frames": source_count,
            "selected_frames": len(selected),
            "repeated_positions": repeats,
            "first_source_frame_index": int(index.source_indices[selected[0]]),
            "last_source_frame_index": int(index.source_indices[selected[-1]]),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(os.path.join(output_dir, "frame_sampling_audit.csv"), index=False)
    return audit


class VideoClipDataset(Dataset):
    def __init__(self, records, index):
        self.records = records.reset_index(drop=True).copy()
        self.index = index
        self.selections = [
            selected_global_indices(index, video_id)[0]
            for video_id in self.records["video_id"].astype(str)
        ]
        self._handles = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _handle(self, path):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > 8:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def __getitem__(self, item):
        row = self.records.iloc[item]
        video_index = self.index.video_lookup[str(row.video_id)]
        path = str(self.index.video_paths[video_index])
        handle = self._handle(path)
        frames = []
        for global_index in self.selections[item]:
            start = int(self.index.starts[global_index])
            end = int(self.index.ends[global_index])
            handle.seek(start)
            encoded = torch.frombuffer(bytearray(handle.read(end - start)), dtype=torch.uint8)
            frame = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
            frames.append(frame)
        return (
            torch.stack(frames),
            torch.tensor(float(row.hemoglobin_g_dl), dtype=torch.float32),
            torch.tensor(item, dtype=torch.long),
        )

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        self.close()
