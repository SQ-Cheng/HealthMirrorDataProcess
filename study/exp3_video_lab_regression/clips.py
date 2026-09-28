"""Index and stream short, temporally contiguous MJPEG clips without frame files."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from io import BytesIO
import json
import mmap
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import ImageReadMode, decode_jpeg

from study.exp2_face_history_head32_regression.frame_index import video_path_for_row

from .config import (
    CLIP_FRAMES, CLIP_POSITIONS, CROP_SIZE, INDEX_DIR,
    MAX_OPEN_FILES_PER_WORKER, SOURCE_SIZE,
)


def _jpeg_ranges(mapped):
    ranges = []
    position = 0
    while True:
        start = mapped.find(b"\xff\xd8", position)
        if start < 0:
            break
        end_marker = mapped.find(b"\xff\xd9", start + 2)
        next_start = mapped.find(b"\xff\xd8", start + 2)
        if end_marker < 0 or 0 <= next_start < end_marker:
            ranges.append(None)
            if next_start < 0:
                break
            position = next_start
            continue
        end = end_marker + 2
        ranges.append((start, end))
        position = end
    return ranges


def _select_clips(mapped, ranges, clip_frames=CLIP_FRAMES, positions=CLIP_POSITIONS):
    if len(ranges) < clip_frames:
        return []
    valid = {}

    def frame_valid(index):
        if index not in valid:
            span = ranges[index]
            if span is None:
                valid[index] = False
            else:
                try:
                    with Image.open(BytesIO(mapped[span[0]:span[1]])) as image:
                        image.load()
                        valid[index] = image.size == (SOURCE_SIZE, SOURCE_SIZE)
                except (OSError, ValueError):
                    valid[index] = False
        return valid[index]

    chosen = []
    last_start = len(ranges) - clip_frames
    for quantile in positions:
        target = round(quantile * last_start)
        for distance in range(last_start + 1):
            options = (target,) if distance == 0 else (target - distance, target + distance)
            selected = None
            for start in options:
                if start < 0 or start > last_start:
                    continue
                if any(start < other + clip_frames and other < start + clip_frames
                       for other in chosen):
                    continue
                if all(frame_valid(i) for i in range(start, start + clip_frames)):
                    selected = start
                    break
            if selected is not None:
                chosen.append(selected)
                break
    return sorted(chosen)


@dataclass
class ClipIndex:
    video_ids: np.ndarray
    video_paths: np.ndarray
    video_ptr: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    source_indices: np.ndarray

    def __post_init__(self):
        self.lookup = {str(video_id): index for index, video_id in enumerate(self.video_ids)}
        if (len(self.video_ptr) != len(self.video_ids) + 1
                or self.starts.shape != self.ends.shape
                or self.starts.shape != self.source_indices.shape
                or self.starts.ndim != 2 or self.starts.shape[1] < 1
                or self.video_ptr[-1] != len(self.starts)
                or not np.all(np.diff(self.source_indices, axis=1) == 1)):
            raise ValueError("Invalid contiguous-clip index")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            return cls(**{name: values[name] for name in (
                "video_ids", "video_paths", "video_ptr", "starts", "ends",
                "source_indices",
            )})

    def clip_range(self, video_id):
        position = self.lookup[str(video_id)]
        return int(self.video_ptr[position]), int(self.video_ptr[position + 1])


def build_or_reuse_index(video_records, index_dir=INDEX_DIR,
                         clip_frames=CLIP_FRAMES, positions=CLIP_POSITIONS):
    index_dir = Path(index_dir)
    index_path = index_dir / "clip_offsets.npz"
    manifest_path = index_dir / "index_manifest.json"
    records = video_records[["video_id", "mirror", "lab_patient_id"]].copy()
    if records.groupby("video_id")[["mirror", "lab_patient_id"]].nunique().gt(1).any().any():
        raise ValueError("Conflicting paths for the same video ID")
    records = records.drop_duplicates("video_id").sort_values("video_id")
    expected_ids = records.video_id.astype(str).tolist()
    if index_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest["schema_version"] == 1
                    and manifest["clip_frames"] == clip_frames
                    and manifest["positions"] == list(positions)
                    and manifest["requested_video_ids"] == expected_ids
                    and all(
                        Path(item["path"]).stat().st_size == item["size_bytes"]
                        and Path(item["path"]).stat().st_mtime_ns == item["mtime_ns"]
                        for item in manifest["source_files"]
                    )):
                print(f"[clip-index-reused] path={index_path}", flush=True)
                return ClipIndex.load(index_path)
        except (KeyError, OSError, ValueError):
            pass

    index_dir.mkdir(parents=True, exist_ok=True)
    video_ids, video_paths, ptr = [], [], [0]
    starts, ends, source_indices, summaries, source_files = [], [], [], [], []
    for number, row in enumerate(records.itertuples(index=False), start=1):
        path = Path(video_path_for_row(row))
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        source_files.append({
            "video_id": str(row.video_id), "path": str(path),
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        })
        windows, frame_count = [], 0
        if stat.st_size:
            with open(path, "rb") as handle:
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    ranges = _jpeg_ranges(mapped)
                    frame_count = len(ranges)
                    windows = _select_clips(mapped, ranges, clip_frames, positions)
                    for start in windows:
                        offsets = ranges[start:start + clip_frames]
                        starts.append([span[0] for span in offsets])
                        ends.append([span[1] for span in offsets])
                        source_indices.append(list(range(start, start + clip_frames)))
        if windows:
            video_ids.append(str(row.video_id))
            video_paths.append(str(path))
            ptr.append(len(starts))
        summaries.append({
            "video_id": str(row.video_id), "source_frames": frame_count,
            "indexed_clips": len(windows), "clip_starts": ";".join(map(str, windows)),
            "status": "indexed" if windows else "excluded_no_valid_contiguous_clip",
        })
        if number % 100 == 0 or number == len(records):
            print(f"[clip-index] scanned={number}/{len(records)} "
                  f"videos={len(video_ids)} clips={len(starts)}", flush=True)
    if not starts:
        raise RuntimeError("No valid contiguous video clips found")
    temporary_path = index_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        video_ids=np.asarray(video_ids, dtype=str),
        video_paths=np.asarray(video_paths, dtype=str),
        video_ptr=np.asarray(ptr, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        source_indices=np.asarray(source_indices, dtype=np.int32),
    )
    os.replace(temporary_path, index_path)
    pd.DataFrame(summaries).to_csv(index_dir / "video_clip_summary.csv", index=False)
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "clip_frames": clip_frames,
        "positions": list(positions), "requested_video_ids": expected_ids,
        "source_files": source_files,
        "storage": "JPEG byte offsets only; decoded frames are never persisted",
    }, indent=2), encoding="utf-8")
    return ClipIndex.load(index_path)


class VideoClipDataset(Dataset):
    def __init__(self, index, records, train=False):
        self.index = index
        self.records = records.reset_index(drop=True).copy()
        self.train = train
        self._handles = OrderedDict()
        self.clip_ids, self.video_rows = [], []
        self.video_clips = []
        for video_row, video_id in enumerate(self.records.video_id.astype(str)):
            first, last = index.clip_range(video_id)
            if first == last:
                raise ValueError(f"No clips for video {video_id}")
            self.video_clips.append(list(range(len(self.clip_ids), len(self.clip_ids) + last - first)))
            self.clip_ids.extend(range(first, last))
            self.video_rows.extend([video_row] * (last - first))
        self.video_rows = np.asarray(self.video_rows, dtype=np.int32)
        self.labels = self.records.robust_scaled_raw_value.to_numpy(np.float32)

    def __len__(self):
        return len(self.clip_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def _handle(self, path):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > MAX_OPEN_FILES_PER_WORKER:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def __getitem__(self, item):
        clip_id = self.clip_ids[item]
        position = int(np.searchsorted(self.index.video_ptr, clip_id, side="right") - 1)
        handle = self._handle(str(self.index.video_paths[position]))
        frames = []
        for start, end in zip(self.index.starts[clip_id], self.index.ends[clip_id]):
            handle.seek(int(start))
            encoded = torch.frombuffer(bytearray(handle.read(int(end - start))), dtype=torch.uint8)
            frame = decode_jpeg(encoded, mode=ImageReadMode.RGB)
            if tuple(frame.shape) != (3, SOURCE_SIZE, SOURCE_SIZE):
                raise RuntimeError(f"Bad decoded frame in clip {clip_id}")
            frames.append(frame)
        clip = torch.stack(frames, dim=1)
        border = (SOURCE_SIZE - CROP_SIZE) // 2
        clip = clip[:, :, border:border + CROP_SIZE, border:border + CROP_SIZE]
        if self.train and torch.rand(()) < 0.5:
            clip = clip.flip(-1)
        video_row = int(self.video_rows[item])
        return clip, torch.tensor(self.labels[video_row]), video_row

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        self.close()


class OneClipPerVideoSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.records)

    def __iter__(self):
        for video_row in torch.randperm(len(self.dataset.video_clips)).tolist():
            candidates = self.dataset.video_clips[video_row]
            yield candidates[int(torch.randint(len(candidates), (1,)).item())]


class PatientDiverseClipBatchSampler(Sampler):
    """Visit each video once while maximizing distinct patients per batch."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patients = dataset.records.hospital_id.astype(str).tolist()

    def __len__(self):
        return (len(self.dataset.records) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        pending = deque(torch.randperm(len(self.dataset.records)).tolist())
        while pending:
            rows, patients = [], set()
            for _ in range(len(pending)):
                if len(rows) == self.batch_size:
                    break
                row = pending.popleft()
                patient = self.patients[row]
                if patient in patients:
                    pending.append(row)
                else:
                    rows.append(row)
                    patients.add(patient)
            while pending and len(rows) < self.batch_size:
                rows.append(pending.popleft())
            batch = []
            for row in rows:
                candidates = self.dataset.video_clips[row]
                batch.append(candidates[int(torch.randint(len(candidates), (1,)).item())])
            yield batch
