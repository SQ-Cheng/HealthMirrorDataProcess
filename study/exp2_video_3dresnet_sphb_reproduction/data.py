"""Exact split reuse and streaming centered contiguous video clips."""

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
    if count < FRAMES_PER_CLIP:
        raise ValueError(
            f"fewer_than_{FRAMES_PER_CLIP}_decodable_frames:{count}"
        )

    source_indices = index.source_indices[start:end]
    boundaries = np.flatnonzero(np.diff(source_indices) != 1) + 1
    runs = np.split(np.arange(count, dtype=np.int64), boundaries)
    candidates = []
    video_midpoint = (float(source_indices[0]) + float(source_indices[-1])) / 2.0
    for run in runs:
        if len(run) < FRAMES_PER_CLIP:
            continue
        first_start = int(run[0])
        last_start = int(run[-1]) - FRAMES_PER_CLIP + 1
        ideal_source_start = video_midpoint - (FRAMES_PER_CLIP - 1) / 2.0
        ideal_local_start = first_start + int(
            np.floor(ideal_source_start - source_indices[first_start])
        )
        candidate_start = min(max(ideal_local_start, first_start), last_start)
        candidate_midpoint = (
            float(source_indices[candidate_start])
            + float(source_indices[candidate_start + FRAMES_PER_CLIP - 1])
        ) / 2.0
        candidates.append(
            (abs(candidate_midpoint - video_midpoint), candidate_start)
        )
    if not candidates:
        longest_run = max((len(run) for run in runs), default=0)
        raise ValueError(
            f"no_{FRAMES_PER_CLIP}_frame_contiguous_run:longest={longest_run}"
        )

    local_start = min(candidates)[1]
    local = np.arange(local_start, local_start + FRAMES_PER_CLIP, dtype=np.int64)
    selected = start + local
    if not np.all(np.diff(index.source_indices[selected]) == 1):
        raise AssertionError(f"Non-contiguous source frame selection for {video_id}")
    return selected, count


def filter_records_for_complete_clips(records, index, output_dir):
    retained_rows, excluded_rows = [], []
    for row_index, row in enumerate(records.itertuples(index=False)):
        try:
            selected_global_indices(index, row.video_id)
            retained_rows.append(row_index)
        except ValueError as error:
            start, end = index.frame_range(row.video_id)
            source_indices = index.source_indices[start:end]
            boundaries = np.flatnonzero(np.diff(source_indices) != 1) + 1
            runs = np.split(np.arange(len(source_indices)), boundaries)
            excluded_rows.append({
                "hospital_id": row.hospital_id,
                "video_id": row.video_id,
                "split": row.split,
                "source_decodable_frames": len(source_indices),
                "longest_contiguous_run": max((len(run) for run in runs), default=0),
                "reason": str(error),
            })
    exclusions = pd.DataFrame(excluded_rows, columns=[
        "hospital_id", "video_id", "split", "source_decodable_frames",
        "longest_contiguous_run", "reason",
    ])
    exclusions.to_csv(
        os.path.join(output_dir, "frame_sampling_exclusions.csv"), index=False
    )
    return records.iloc[retained_rows].reset_index(drop=True), exclusions


def write_sampling_audit(records, index, output_dir):
    rows = []
    for row in records.itertuples(index=False):
        selected, source_count = selected_global_indices(index, row.video_id)
        selected_sources = index.source_indices[selected]
        selected_center = (
            float(selected_sources[0]) + float(selected_sources[-1])
        ) / 2.0
        source_start, source_end = index.frame_range(row.video_id)
        available_sources = index.source_indices[source_start:source_end]
        available_center = (
            float(available_sources[0]) + float(available_sources[-1])
        ) / 2.0
        rows.append({
            "hospital_id": row.hospital_id,
            "video_id": row.video_id,
            "split": row.split,
            "source_decodable_frames": source_count,
            "selected_frames": len(selected),
            "sampling_policy": "centered_contiguous_window",
            "first_source_frame_index": int(selected_sources[0]),
            "last_source_frame_index": int(selected_sources[-1]),
            "source_frame_span": int(selected_sources[-1] - selected_sources[0] + 1),
            "maximum_source_frame_gap": int(np.diff(selected_sources).max()),
            "center_offset_from_available_frames": selected_center - available_center,
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
