"""Exact parent-window reuse with native-resolution video-only loading."""

from collections import OrderedDict
import hashlib
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .config import (
    EXPECTED_ECG_MAX_GAP_SECONDS,
    EXPECTED_ECG_SAMPLE_RATE_HZ,
    EXPECTED_PARENT_TASK_TYPE,
    HORIZONTAL_FLIP_PROBABILITY,
    TARGETS,
    TIMESTAMP_CACHE_RECORDINGS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


WINDOW_COLUMNS = {
    "target",
    "window_id",
    "video_id",
    "hospital_id",
    "split",
    "target_score",
    "video_path",
    "video_times_path",
    "frame_start_index",
    "frame_end_index",
    "video_frame_count",
    "ecg_max_source_gap_seconds",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_parent_contract(parent_output_dir, targets, require_complete):
    experiment_path = os.path.join(parent_output_dir, "experiment_manifest.json")
    dataset_path = os.path.join(parent_output_dir, "dataset_manifest.json")
    score_path = os.path.join(parent_output_dir, "score_definition.json")
    for path in (experiment_path, dataset_path, score_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing parent experiment artifact: {path}")
    experiment = _load_json(experiment_path)
    dataset = _load_json(dataset_path)
    score = _load_json(score_path)
    if experiment.get("task_type") != EXPECTED_PARENT_TASK_TYPE:
        raise RuntimeError("Parent is not the expected abnormal-score regression")
    ecg = experiment.get("inputs", {}).get("ecg_resampling", {})
    if int(ecg.get("sample_rate_hz", -1)) != EXPECTED_ECG_SAMPLE_RATE_HZ:
        raise RuntimeError("Parent ECG sample rate differs from paired contract")
    if not np.isclose(
        float(ecg.get("max_source_gap_seconds", np.nan)),
        EXPECTED_ECG_MAX_GAP_SECONDS,
        atol=1e-12,
    ):
        raise RuntimeError("Parent ECG gap threshold differs from paired contract")
    if experiment.get("inputs", {}).get("video_interpolation") is not False:
        raise RuntimeError("Parent video interpolation contract changed")
    if experiment.get("inputs", {}).get("video_frame_sampling") is not False:
        raise RuntimeError("Parent video frame-sampling contract changed")
    missing_targets = sorted(set(targets) - set(experiment.get("targets", [])))
    if missing_targets:
        raise RuntimeError(f"Parent experiment lacks targets: {missing_targets}")
    if score.get("transform") != "asinh":
        raise RuntimeError("Parent abnormal-score transform changed")
    if require_complete:
        run_index_path = os.path.join(parent_output_dir, "run_index.csv")
        if not os.path.isfile(run_index_path):
            raise RuntimeError("Parent run_index.csv is missing")
        run_index = pd.read_csv(run_index_path)
        if run_index["target"].duplicated().any():
            raise RuntimeError("Parent run_index.csv contains duplicate targets")
        indexed = set(run_index.loc[run_index["status"].eq("ok"), "target"])
        missing = sorted(set(TARGETS) - indexed)
        failed = run_index.loc[run_index["status"].ne("ok")]
        if missing or not failed.empty:
            raise RuntimeError(
                f"Parent training is not complete: missing={missing}, "
                f"non_ok={failed['target'].tolist()}"
            )
    return experiment, dataset, score


def _validate_windows(frame, target):
    missing = sorted(WINDOW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{target} parent windows lack columns: {missing}")
    if frame.empty or frame["window_id"].duplicated().any():
        raise ValueError(f"{target} has empty or duplicate parent windows")
    if not frame["target"].eq(target).all():
        raise ValueError(f"{target} window file contains another target")
    if not set(frame["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError(f"{target} has invalid split values")
    if set(frame["split"].unique()) != {"train", "val", "test"}:
        raise ValueError(f"{target} does not contain all three splits")
    scores = pd.to_numeric(frame["target_score"], errors="coerce")
    if not np.isfinite(scores).all():
        raise ValueError(f"{target} has non-finite abnormal scores")
    gaps = pd.to_numeric(frame["ecg_max_source_gap_seconds"], errors="coerce")
    if (
        not np.isfinite(gaps).all()
        or gaps.gt(EXPECTED_ECG_MAX_GAP_SECONDS + 1e-9).any()
    ):
        raise ValueError(f"{target} violates paired ECG quality filtering")
    patient_sets = {
        split: set(
            frame.loc[frame["split"].eq(split), "hospital_id"].astype(str)
        )
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if patient_sets[left] & patient_sets[right]:
            raise ValueError(f"{target} patient leakage: {left}/{right}")
    counts = frame.groupby("video_id")["target_score"].nunique()
    if counts.gt(1).any():
        raise ValueError(f"{target} has inconsistent scores within a video")


def prepare_ablation_data(
    parent_output_dir,
    output_dir,
    targets=TARGETS,
    require_parent_complete=True,
):
    experiment, dataset, score = _validate_parent_contract(
        parent_output_dir, targets, require_parent_complete
    )
    windows_dir = os.path.join(output_dir, "windows")
    snapshot_dir = os.path.join(output_dir, "parent_snapshot")
    os.makedirs(windows_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    hash_rows, task_frames = [], {}
    for target in targets:
        source = os.path.join(parent_output_dir, "windows", f"{target}.csv")
        destination = os.path.join(windows_dir, f"{target}.csv")
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Missing parent windows: {source}")
        frame = pd.read_csv(
            source,
            dtype={"hospital_id": str, "video_id": str, "window_id": str},
        )
        _validate_windows(frame, target)
        shutil.copy2(source, destination)
        source_hash, copied_hash = _sha256(source), _sha256(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"Window snapshot hash mismatch for {target}")
        hash_rows.append(
            {
                "target": target,
                "source_path": os.path.abspath(source),
                "snapshot_path": os.path.abspath(destination),
                "sha256": source_hash,
                "rows": int(len(frame)),
                "videos": int(frame["video_id"].nunique()),
            }
        )
        task_frames[target] = frame

    snapshot_files = (
        "dataset_manifest.json",
        "experiment_manifest.json",
        "score_definition.json",
        "split_assignment_manifest.json",
        "task_summary.csv",
        "split_distribution_audit.csv",
        "split_distribution_pairwise.csv",
        "split_distribution.png",
    )
    for name in snapshot_files:
        source = os.path.join(parent_output_dir, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(snapshot_dir, name))
    shutil.copy2(
        os.path.join(parent_output_dir, "score_definition.json"),
        os.path.join(output_dir, "score_definition.json"),
    )
    if require_parent_complete:
        shutil.copy2(
            os.path.join(parent_output_dir, "run_index.csv"),
            os.path.join(snapshot_dir, "parent_run_index.csv"),
        )
    hash_frame = pd.DataFrame(hash_rows)
    hash_frame.to_csv(os.path.join(output_dir, "window_hashes.csv"), index=False)
    with open(
        os.path.join(output_dir, "ablation_data_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "ablation": "remove_ecg_input_only",
                "parent_output_dir": os.path.abspath(parent_output_dir),
                "exact_parent_window_rows": True,
                "exact_parent_patient_splits": True,
                "exact_parent_target_scores": True,
                "exact_parent_video_frame_boundaries": True,
                "parent_completion_required": bool(require_parent_complete),
                "parent_experiment": experiment.get("experiment"),
                "parent_dataset_schema": dataset.get("schema_version"),
                "score_transform": score.get("transform"),
                "targets": list(targets),
            },
            handle,
            indent=2,
        )
    return task_frames, hash_frame


def _read_video_times(path):
    frame = pd.read_csv(path)
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"Invalid video timestamps: {path}")
    return values


class VideoOnlyWindowDataset(Dataset):
    def __init__(self, windows, training=False):
        self.windows = windows.reset_index(drop=True).copy()
        self.training = bool(training)
        self._timestamp_cache = OrderedDict()

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_timestamp_cache"] = OrderedDict()
        return state

    def _video_times(self, path):
        value = self._timestamp_cache.pop(path, None)
        if value is None:
            value = _read_video_times(path)
        self._timestamp_cache[path] = value
        while len(self._timestamp_cache) > TIMESTAMP_CACHE_RECORDINGS:
            self._timestamp_cache.popitem(last=False)
        return value

    @staticmethod
    def _decode_frames(path, start, end):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        if not seek_ok and start:
            capture.release()
            capture = cv2.VideoCapture(path)
            for _ in range(int(start)):
                if not capture.grab():
                    capture.release()
                    raise RuntimeError(f"Cannot seek to frame {start}: {path}")
        frames = []
        for frame_index in range(int(start), int(end)):
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise RuntimeError(
                    f"Decode stopped at frame {frame_index}, expected {end}: {path}"
                )
            if tuple(frame.shape) != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                capture.release()
                raise RuntimeError(f"Unexpected frame shape {frame.shape}: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, index):
        row = self.windows.iloc[int(index)]
        frame_start = int(row.frame_start_index)
        frame_end = int(row.frame_end_index)
        frames = self._decode_frames(row.video_path, frame_start, frame_end)
        times = self._video_times(row.video_times_path)[frame_start:frame_end]
        if len(times) != len(frames):
            raise RuntimeError(f"Frame/timestamp mismatch for {row.window_id}")
        relative_times = torch.from_numpy(
            (times - float(row.window_start_unix)).astype(np.float32)
        )
        if self.training and torch.rand(()) < HORIZONTAL_FLIP_PROBABILITY:
            frames = torch.flip(frames, dims=(-1,))
        return {
            "frames": frames,
            "frame_times": relative_times,
            "target_score": torch.tensor(
                float(row.target_score), dtype=torch.float32
            ),
            "window_id": str(row.window_id),
            "video_id": str(row.video_id),
            "hospital_id": str(row.hospital_id),
        }


def collate_windows(samples):
    return {
        "frames": pad_sequence(
            [sample["frames"] for sample in samples],
            batch_first=True,
            padding_value=0,
        ),
        "frame_times": pad_sequence(
            [sample["frame_times"] for sample in samples],
            batch_first=True,
            padding_value=0.0,
        ),
        "frame_lengths": torch.tensor(
            [len(sample["frames"]) for sample in samples], dtype=torch.long
        ),
        "targets": torch.stack([sample["target_score"] for sample in samples]),
        "window_ids": [sample["window_id"] for sample in samples],
        "video_ids": [sample["video_id"] for sample in samples],
        "hospital_ids": [sample["hospital_id"] for sample in samples],
    }


__all__ = [
    "VideoOnlyWindowDataset",
    "collate_windows",
    "prepare_ablation_data",
]
