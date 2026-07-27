"""Conflict-safe video tasks and streaming all-frame image loading."""

from collections import OrderedDict
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Sampler
from torchvision.io import ImageReadMode, decode_jpeg

from .config import (
    DECODE_CACHE_FRAMES,
    FRAME_SHUFFLE_CHUNK_SIZE,
    MAX_OPEN_FILES_PER_WORKER,
    MIN_PATIENTS_PER_CLASS,
    MIN_VIDEOS_PER_CLASS,
    SEED,
    SOURCE_IMAGE_SIZE,
    TARGETS,
)


def validate_source_data(source_dir):
    required = ("base_manifest.csv", "data_quality_report.json")
    missing = [name for name in required if not os.path.exists(os.path.join(source_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing corrected Exp2 source files: {missing}")
    with open(
        os.path.join(source_dir, "data_quality_report.json"), encoding="utf-8"
    ) as handle:
        quality = json.load(handle)
    timezone = quality.get("lab_report_time", {}).get("source_timezone")
    if timezone != "Asia/Shanghai":
        raise RuntimeError(f"Corrected Asia/Shanghai time basis is missing: {timezone}")
    policy = quality.get("hemoglobin_conflicting_video_policy", {})
    if policy.get("scope") != "per target" or "hemoglobin_low" not in policy.get("targets", []):
        raise RuntimeError("Corrected Hb conflict policy is missing")
    return quality


def _conflicting_video_audit(events, target):
    rows = []
    for video_id, group in events.groupby("video_id", sort=True):
        labels = pd.to_numeric(group[target], errors="coerce").dropna().astype(int)
        positive_count = int(labels.eq(1).sum())
        negative_count = int(labels.eq(0).sum())
        if positive_count and negative_count:
            rows.append({
                "target": target,
                "video_id": str(video_id),
                "hospital_id": str(group["hospital_id"].iloc[0]),
                "positive_event_count": positive_count,
                "negative_event_count": negative_count,
                "event_count": int(len(group)),
                "action": "excluded_for_this_target",
            })
    return rows


def build_task_records(base_manifest, target):
    labels = pd.to_numeric(base_manifest[target], errors="coerce")
    invalid_labels = sorted(set(labels.dropna().unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Non-binary values for {target}: {invalid_labels}")
    events = base_manifest.loc[labels.notna()].copy()
    events[target] = labels.loc[events.index].astype(int)
    if events.empty:
        return pd.DataFrame(), [], {
            "target": target, "status": "skipped", "reason": "no non-missing labels"
        }
    audit_rows = _conflicting_video_audit(events, target)
    conflicting_videos = {row["video_id"] for row in audit_rows}
    clean = events.loc[~events["video_id"].astype(str).isin(conflicting_videos)].copy()
    if not clean.groupby("video_id")[target].nunique().le(1).all():
        raise AssertionError(f"Conflicting labels remain for {target}")
    records = clean.sort_values(
        ["hospital_id", "video_id", "sample_id"]
    ).drop_duplicates("video_id", keep="first")
    records = records[
        ["hospital_id", "video_id", "mirror", "lab_patient_id", target]
    ].rename(columns={target: "label"}).reset_index(drop=True)
    positive_videos = int(records["label"].eq(1).sum())
    negative_videos = int(records["label"].eq(0).sum())
    summary = {
        "target": target,
        "status": "pending",
        "reason": "",
        "source_events": int(len(events)),
        "source_videos": int(events["video_id"].nunique()),
        "conflicting_videos": int(len(conflicting_videos)),
        "clean_videos": int(len(records)),
        "clean_patients": int(records["hospital_id"].astype(str).nunique()),
        "positive_videos": positive_videos,
        "negative_videos": negative_videos,
    }
    if min(positive_videos, negative_videos) < MIN_VIDEOS_PER_CLASS:
        summary.update({
            "status": "skipped",
            "reason": (
                f"fewer than {MIN_VIDEOS_PER_CLASS} clean videos in one class: "
                f"positive={positive_videos}, negative={negative_videos}"
            ),
        })
    return records, audit_rows, summary


def add_patient_split(records, target, seed=SEED):
    patient_labels = records.groupby("hospital_id")["label"].max().reset_index()
    class_counts = patient_labels["label"].value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < MIN_PATIENTS_PER_CLASS:
        return None, (
            f"fewer than {MIN_PATIENTS_PER_CLASS} stratifiable patients in one class: "
            f"{class_counts.to_dict()}"
        )
    try:
        first = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
        train_index, temporary_index = next(
            first.split(patient_labels["hospital_id"], patient_labels["label"])
        )
        temporary = patient_labels.iloc[temporary_index].reset_index(drop=True)
        second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
        validation_index, test_index = next(
            second.split(temporary["hospital_id"], temporary["label"])
        )
    except ValueError as exc:
        return None, f"patient stratification failed: {exc}"
    split_by_patient = {}
    for split_name, frame in (
        ("train", patient_labels.iloc[train_index]),
        ("val", temporary.iloc[validation_index]),
        ("test", temporary.iloc[test_index]),
    ):
        split_by_patient.update({str(value): split_name for value in frame["hospital_id"]})
    result = records.copy()
    result["hospital_id"] = result["hospital_id"].astype(str)
    result["split"] = result["hospital_id"].map(split_by_patient)
    if result["split"].isna().any():
        raise AssertionError(f"Unassigned patients for {target}")
    patient_sets = {
        split: set(result.loc[result["split"].eq(split), "hospital_id"])
        for split in ("train", "val", "test")
    }
    if any(patient_sets[a] & patient_sets[b] for a, b in (
        ("train", "val"), ("train", "test"), ("val", "test")
    )):
        raise AssertionError(f"Patient leakage detected for {target}")
    return result.reset_index(drop=True), ""


def prepare_tasks(base_manifest, output_dir, targets=TARGETS, seed=SEED):
    os.makedirs(os.path.join(output_dir, "task_records"), exist_ok=True)
    summaries, conflicts, task_records = [], [], {}
    for target in targets:
        records, audit_rows, summary = build_task_records(base_manifest, target)
        conflicts.extend(audit_rows)
        if summary["status"] != "skipped":
            records, reason = add_patient_split(records, target, seed)
            if records is None:
                summary.update({"status": "skipped", "reason": reason})
            else:
                summary["status"] = "ready"
                for split in ("train", "val", "test"):
                    summary[f"{split}_patients"] = int(
                        records.loc[records["split"].eq(split), "hospital_id"].nunique()
                    )
                    summary[f"{split}_videos"] = int(records["split"].eq(split).sum())
                records.to_csv(
                    os.path.join(output_dir, "task_records", f"{target}.csv"), index=False
                )
                task_records[target] = records
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    conflict_frame = pd.DataFrame(conflicts)
    summary_frame.to_csv(os.path.join(output_dir, "task_summary.csv"), index=False)
    conflict_frame.to_csv(os.path.join(output_dir, "conflicting_videos.csv"), index=False)
    return task_records, summary_frame, conflict_frame


class GroupedFrameViewSampler(Sampler):
    """Shuffle contiguous frame chunks while keeping each frame's views together."""

    def __init__(self, dataset, chunk_size=FRAME_SHUFFLE_CHUNK_SIZE):
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    def __iter__(self):
        frame_count = self.dataset.frame_count
        chunk_count = (frame_count + self.chunk_size - 1) // self.chunk_size
        for chunk_index in torch.randperm(chunk_count).tolist():
            start = chunk_index * self.chunk_size
            end = min(start + self.chunk_size, frame_count)
            for frame_index in range(start, end):
                if self.dataset.expand_all_views:
                    yield frame_index
                else:
                    sample_offset = frame_index * len(self.dataset.views)
                    for view_index in torch.randperm(len(self.dataset.views)).tolist():
                        yield sample_offset + view_index

    def __len__(self):
        return len(self.dataset)


class AllFramesDataset(Dataset):
    def __init__(
        self,
        frame_index,
        video_records,
        views=("original",),
        interpolation="bilinear",
        expand_all_views=False,
    ):
        self.index = frame_index
        self.video_records = video_records.reset_index(drop=True).copy()
        self.views = tuple(views)
        self.interpolation = interpolation
        self.expand_all_views = bool(expand_all_views)
        if self.expand_all_views and len(self.views) < 2:
            raise ValueError("expand_all_views requires multiple training views")
        frame_indices, frame_video_rows = [], []
        record_index_videos = []
        for record_index, row in enumerate(self.video_records.itertuples(index=False)):
            index_video = self.index.video_lookup[str(row.video_id)]
            start, end = self.index.frame_range(row.video_id)
            frame_indices.append(np.arange(start, end, dtype=np.int64))
            frame_video_rows.append(np.full(end - start, record_index, dtype=np.int32))
            record_index_videos.append(index_video)
        self.frame_indices = np.concatenate(frame_indices)
        self.frame_video_rows = np.concatenate(frame_video_rows)
        self.record_index_videos = np.asarray(record_index_videos, dtype=np.int32)
        self.labels_by_video = self.video_records["label"].to_numpy(np.float32)
        self.frame_count = len(self.frame_indices)
        self._handles = OrderedDict()
        self._decoded_cache = OrderedDict()

    def __len__(self):
        if self.expand_all_views:
            return self.frame_count
        return self.frame_count * len(self.views)

    @property
    def model_input_count(self):
        return self.frame_count * len(self.views)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_decoded_cache"] = OrderedDict()
        return state

    def _handle(self, path):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > MAX_OPEN_FILES_PER_WORKER:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def _decode(self, global_frame_index, video_row):
        cached = self._decoded_cache.pop(global_frame_index, None)
        if cached is not None:
            self._decoded_cache[global_frame_index] = cached
            return cached
        index_video = self.record_index_videos[video_row]
        path = str(self.index.video_paths[index_video])
        start = int(self.index.starts[global_frame_index])
        end = int(self.index.ends[global_frame_index])
        handle = self._handle(path)
        handle.seek(start)
        payload = handle.read(end - start)
        try:
            encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            tensor = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
            expected_shape = (3, SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"unexpected decoded shape {tuple(tensor.shape)}")
        except Exception as exc:
            raise RuntimeError(
                f"Decode failed path={path} frame={global_frame_index} "
                f"offsets={start}:{end}: {exc}"
            ) from exc
        self._decoded_cache[global_frame_index] = tensor
        while len(self._decoded_cache) > DECODE_CACHE_FRAMES:
            self._decoded_cache.popitem(last=False)
        return tensor

    def __getitem__(self, index):
        if self.expand_all_views:
            base_frame_index = index
            view_index = torch.arange(len(self.views), dtype=torch.uint8)
        else:
            base_frame_index = index // len(self.views)
            view_index = torch.tensor(index % len(self.views), dtype=torch.uint8)
        global_frame_index = int(self.frame_indices[base_frame_index])
        video_row = int(self.frame_video_rows[base_frame_index])
        image = self._decode(global_frame_index, video_row)
        label = self.labels_by_video[video_row]
        return (
            image,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(base_frame_index, dtype=torch.long),
            view_index,
        )

    def frame_labels(self):
        return self.labels_by_video[self.frame_video_rows]

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._decoded_cache.clear()

    def __del__(self):
        self.close()
