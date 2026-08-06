"""Timestamp-aligned native video and uniformly resampled ECG windows."""

from collections import OrderedDict
import json
import math
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from study.exp2_face_pretrained_head32_regression.data import (
    _plot_split_distributions,
    add_patient_split,
    build_task_records,
    validate_source_data,
)

from .config import (
    ECG_CACHE_RECORDINGS,
    ECG_MAX_INTERPOLATION_GAP_SECONDS,
    ECG_SAMPLE_RATE_HZ,
    ECG_SAMPLES_PER_WINDOW,
    HORIZONTAL_FLIP_PROBABILITY,
    LAB_TARGET_PREFIXES,
    MIN_ECG_SAMPLES_PER_WINDOW,
    MIN_VIDEO_FRAMES_PER_WINDOW,
    RAW_DATA_ROOT,
    SEED,
    TARGETS,
    TIMESTAMP_CACHE_RECORDINGS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)


def validate_lab_time_alignment(manifest, targets=TARGETS):
    for target in targets:
        prefix = LAB_TARGET_PREFIXES.get(target)
        if prefix is None:
            continue
        labelled = pd.to_numeric(manifest[target], errors="coerce").notna()
        delta = pd.to_numeric(
            manifest.loc[labelled, f"{prefix}_delta_h"], errors="coerce"
        )
        signed = pd.to_numeric(
            manifest.loc[labelled, f"{prefix}_signed_delta_h"], errors="coerce"
        )
        invalid = (
            delta.isna()
            | signed.isna()
            | delta.lt(0.0)
            | delta.gt(24.0 + 1e-6)
            | ~np.isclose(delta.to_numpy(), np.abs(signed.to_numpy()), atol=1e-7)
        )
        if invalid.any():
            raise ValueError(
                f"Invalid corrected 24-hour alignment for {target}: "
                f"{int(invalid.sum())}"
            )


def _read_video_times(path):
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError("video timestamp file has fewer than two columns")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("video timestamps contain non-finite values")
    return values


def _read_ecg(path):
    frame = pd.read_csv(path, header=None, usecols=[0, 1])
    times = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(np.float64)
    signal = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(np.float32)
    valid = np.isfinite(times) & np.isfinite(signal)
    return times[valid], signal[valid]


def _recording_paths(raw_root, mirror, patient_id):
    directory = os.path.join(
        raw_root, f"{mirror}_data", f"patient_{int(patient_id):06d}"
    )
    return {
        "raw_directory": directory,
        "video_path": os.path.join(directory, "video.avi"),
        "video_times_path": os.path.join(directory, "video.avi.ts"),
        "ecg_path": os.path.join(directory, "ecg_log.csv"),
    }


def _window_rows(video_id, video_times, ecg_times, overlap_start, overlap_end):
    duration = overlap_end - overlap_start
    count = int(
        math.floor(
            (duration - WINDOW_SECONDS + 1e-9) / WINDOW_STRIDE_SECONDS
        )
        + 1
    )
    rows = []
    stats = {
        "candidate_window_count": int(max(count, 0)),
        "insufficient_samples_window_count": 0,
        "ecg_bracketing_failure_window_count": 0,
        "ecg_gap_rejected_window_count": 0,
    }
    for window_index in range(max(count, 0)):
        start = overlap_start + window_index * WINDOW_STRIDE_SECONDS
        end = start + WINDOW_SECONDS
        frame_start = int(np.searchsorted(video_times, start, side="left"))
        frame_end = int(np.searchsorted(video_times, end, side="left"))
        raw_ecg_start = int(np.searchsorted(ecg_times, start, side="left"))
        raw_ecg_end = int(np.searchsorted(ecg_times, end, side="left"))
        frame_count = frame_end - frame_start
        ecg_count = raw_ecg_end - raw_ecg_start
        if (
            frame_count < MIN_VIDEO_FRAMES_PER_WINDOW
            or ecg_count < MIN_ECG_SAMPLES_PER_WINDOW
        ):
            stats["insufficient_samples_window_count"] += 1
            continue
        final_resample_time = start + (
            (ECG_SAMPLES_PER_WINDOW - 1) / ECG_SAMPLE_RATE_HZ
        )
        interpolation_start = max(
            int(np.searchsorted(ecg_times, start, side="right")) - 1,
            0,
        )
        interpolation_right = int(
            np.searchsorted(ecg_times, final_resample_time, side="left")
        )
        if interpolation_right >= len(ecg_times):
            stats["ecg_bracketing_failure_window_count"] += 1
            continue
        interpolation_end = interpolation_right + 1
        interpolation_times = ecg_times[interpolation_start:interpolation_end]
        if (
            len(interpolation_times) < 2
            or interpolation_times[0] > start
            or interpolation_times[-1] < final_resample_time
        ):
            stats["ecg_bracketing_failure_window_count"] += 1
            continue
        max_gap = float(np.max(np.diff(interpolation_times)))
        if max_gap > ECG_MAX_INTERPOLATION_GAP_SECONDS:
            stats["ecg_gap_rejected_window_count"] += 1
            continue
        rows.append(
            {
                "window_id": f"{video_id}_w{window_index:02d}",
                "video_id": video_id,
                "window_index": window_index,
                "window_start_unix": start,
                "window_end_unix": end,
                "frame_start_index": frame_start,
                "frame_end_index": frame_end,
                "video_frame_count": frame_count,
                "ecg_start_index": interpolation_start,
                "ecg_end_index": interpolation_end,
                "ecg_raw_window_sample_count": ecg_count,
                "ecg_resampled_sample_count": ECG_SAMPLES_PER_WINDOW,
                "ecg_max_source_gap_seconds": max_gap,
            }
        )
    stats["accepted_window_count"] = int(len(rows))
    return rows, stats


def build_recording_index(candidate_records, raw_root=RAW_DATA_ROOT):
    identity = (
        candidate_records[["video_id", "mirror", "lab_patient_id"]]
        .drop_duplicates("video_id")
        .sort_values("video_id")
        .reset_index(drop=True)
    )
    quality_rows, valid_recordings, window_rows = [], [], []
    for index, row in enumerate(identity.itertuples(index=False), start=1):
        video_id = str(row.video_id)
        expected_video_id = (
            f"{row.mirror}_patient_{int(row.lab_patient_id):06d}"
        )
        paths = _recording_paths(raw_root, row.mirror, row.lab_patient_id)
        quality = {
            "video_id": video_id,
            "mirror": row.mirror,
            "lab_patient_id": int(row.lab_patient_id),
            **paths,
            "status": "excluded",
            "reason": "",
        }
        try:
            if video_id != expected_video_id:
                raise ValueError(
                    f"video_id/path identity mismatch: expected {expected_video_id}"
                )
            for key in ("video_path", "video_times_path", "ecg_path"):
                if not os.path.isfile(paths[key]) or os.path.getsize(paths[key]) == 0:
                    raise ValueError(f"missing_or_empty_{key}")
            video_times = _read_video_times(paths["video_times_path"])
            ecg_times, ecg_signal = _read_ecg(paths["ecg_path"])
            if len(video_times) < 2 or len(ecg_times) < 2:
                raise ValueError("recording_has_too_few_samples")
            if np.any(np.diff(video_times) <= 0):
                raise ValueError("video_timestamps_not_strictly_increasing")
            if np.any(np.diff(ecg_times) <= 0):
                raise ValueError("ecg_timestamps_not_strictly_increasing")

            capture = cv2.VideoCapture(paths["video_path"])
            opened, first_frame = capture.isOpened(), capture.read()[1]
            last_seek_ok = capture.set(
                cv2.CAP_PROP_POS_FRAMES, int(len(video_times) - 1)
            )
            last_ok, last_frame = capture.read()
            capture.release()
            if not opened or first_frame is None:
                raise ValueError("video_first_frame_decode_failed")
            if not last_seek_ok or not last_ok or last_frame is None:
                raise ValueError("video_last_timestamped_frame_decode_failed")
            for frame in (first_frame, last_frame):
                if tuple(frame.shape) != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                    raise ValueError(f"unexpected_video_shape_{tuple(frame.shape)}")

            overlap_start = max(float(video_times[0]), float(ecg_times[0]))
            overlap_end = min(float(video_times[-1]), float(ecg_times[-1]))
            windows, window_stats = _window_rows(
                video_id, video_times, ecg_times, overlap_start, overlap_end
            )
            quality.update(window_stats)
            if not windows:
                raise ValueError("no_resampleable_quality_window")
            video_duration = float(video_times[-1] - video_times[0])
            ecg_duration = float(ecg_times[-1] - ecg_times[0])
            recording = {
                **quality,
                "status": "ready",
                "reason": "",
                "video_frame_count": int(len(video_times)),
                "ecg_sample_count": int(len(ecg_times)),
                "video_start_unix": float(video_times[0]),
                "video_end_unix": float(video_times[-1]),
                "ecg_start_unix": float(ecg_times[0]),
                "ecg_end_unix": float(ecg_times[-1]),
                "overlap_start_unix": overlap_start,
                "overlap_end_unix": overlap_end,
                "overlap_seconds": overlap_end - overlap_start,
                "observed_video_fps": (
                    (len(video_times) - 1) / video_duration
                    if video_duration > 0
                    else np.nan
                ),
                "observed_ecg_hz": (
                    (len(ecg_times) - 1) / ecg_duration
                    if ecg_duration > 0
                    else np.nan
                ),
                "max_ecg_source_gap_seconds": float(np.max(np.diff(ecg_times))),
                "ecg_signal_min": float(np.min(ecg_signal)),
                "ecg_signal_max": float(np.max(ecg_signal)),
                "window_count": int(len(windows)),
                **window_stats,
            }
            valid_recordings.append(recording)
            for window in windows:
                window_rows.append({**window, **paths})
            quality.update(recording)
        except Exception as exc:
            quality["reason"] = str(exc)
        quality_rows.append(quality)
        if index % 100 == 0:
            print(f"[recording-index] scanned={index}/{len(identity)}", flush=True)

    quality_frame = pd.DataFrame(quality_rows)
    recordings = pd.DataFrame(valid_recordings)
    windows = pd.DataFrame(window_rows)
    if recordings.empty or windows.empty:
        raise RuntimeError("No valid synchronized video/ECG recordings")
    if windows["window_id"].duplicated().any():
        raise AssertionError("Duplicate synchronized window IDs")
    if not windows.groupby("video_id")["window_index"].apply(
        lambda values: values.is_unique
    ).all():
        raise AssertionError("Duplicate window index within a video")
    return quality_frame, recordings, windows


def prepare_experiment_data(
    source_dir,
    raw_root,
    output_dir,
    targets=TARGETS,
    seed=SEED,
):
    os.makedirs(os.path.join(output_dir, "task_records"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "windows"), exist_ok=True)
    source_quality = validate_source_data(source_dir)
    base_manifest = pd.read_csv(
        os.path.join(source_dir, "base_manifest.csv"),
        dtype={"hospital_id": str},
    )
    video_summary = pd.read_csv(
        os.path.join(source_dir, "video_summary.csv"),
        dtype={"hospital_id": str},
    )
    validate_lab_time_alignment(base_manifest, targets)

    clean_records, summaries, conflicts = {}, [], []
    for target in targets:
        records, audit_rows, summary = build_task_records(
            base_manifest, video_summary, target
        )
        clean_records[target] = records
        summaries.append(summary)
        conflicts.extend(audit_rows)
    candidate_records = pd.concat(
        [
            records[["video_id", "mirror", "lab_patient_id"]]
            for records in clean_records.values()
            if not records.empty
        ],
        ignore_index=True,
    )
    quality, recordings, base_windows = build_recording_index(
        candidate_records, raw_root
    )
    valid_video_ids = set(recordings["video_id"].astype(str))

    task_records = {}
    distribution_rows, pair_rows, selections = [], [], []
    task_window_frames = []
    summary_by_target = {row["target"]: row for row in summaries}
    for target in targets:
        summary = summary_by_target[target]
        records = clean_records[target]
        before_raw_filter = len(records)
        records = records.loc[
            records["video_id"].astype(str).isin(valid_video_ids)
        ].reset_index(drop=True)
        summary["raw_recording_excluded_videos"] = int(before_raw_filter - len(records))
        if records.empty:
            summary.update(
                {"status": "skipped", "reason": "no valid synchronized recordings"}
            )
            continue
        split_records, reason, audit, pairs, selection = add_patient_split(
            records, target, seed
        )
        if split_records is None:
            summary.update({"status": "skipped", "reason": reason})
            selections.append(selection)
            continue
        split_records["target_score"] = pd.to_numeric(
            split_records["abnormal_score"], errors="raise"
        ).astype(np.float32)
        if not np.isfinite(split_records["target_score"]).all():
            raise ValueError(f"Non-finite abnormal scores for {target}")
        split_records.to_csv(
            os.path.join(output_dir, "task_records", f"{target}.csv"), index=False
        )
        windows = split_records.merge(
            base_windows,
            on="video_id",
            how="left",
            validate="one_to_many",
        )
        if windows["window_id"].isna().any():
            raise AssertionError(f"Missing synchronized windows for {target}")
        windows.insert(0, "target", target)
        windows = windows.sort_values(
            ["split", "video_id", "window_index"]
        ).reset_index(drop=True)
        windows.to_csv(
            os.path.join(output_dir, "windows", f"{target}.csv"), index=False
        )
        task_window_frames.append(windows)
        task_records[target] = split_records
        distribution_rows.extend(audit)
        pair_rows.extend(pairs)
        selections.append(selection)
        summary["status"] = "ready"
        summary["reason"] = ""
        summary["synchronized_videos"] = int(len(split_records))
        summary["synchronized_windows"] = int(len(windows))
        for split in ("train", "val", "test"):
            subset = split_records.loc[split_records["split"].eq(split)]
            split_windows = windows.loc[windows["split"].eq(split)]
            summary[f"{split}_patients"] = int(subset["hospital_id"].nunique())
            summary[f"{split}_videos"] = int(len(subset))
            summary[f"{split}_windows"] = int(len(split_windows))
            summary[f"{split}_abnormal_windows"] = int(
                split_windows["target_score"].gt(0).sum()
            )
            summary[f"{split}_normal_windows"] = int(
                split_windows["target_score"].lt(0).sum()
            )

    if not task_records:
        raise RuntimeError("All requested tasks were skipped")
    summary_frame = pd.DataFrame(summaries)
    conflict_frame = pd.DataFrame(
        conflicts,
        columns=(
            "target",
            "video_id",
            "hospital_id",
            "positive_event_count",
            "negative_event_count",
            "event_count",
            "action",
        ),
    )
    quality.to_csv(os.path.join(output_dir, "recording_quality.csv"), index=False)
    recordings.to_csv(os.path.join(output_dir, "recordings.csv"), index=False)
    base_windows.to_csv(os.path.join(output_dir, "base_windows.csv"), index=False)
    summary_frame.to_csv(os.path.join(output_dir, "task_summary.csv"), index=False)
    conflict_frame.to_csv(
        os.path.join(output_dir, "conflicting_videos.csv"), index=False
    )
    pd.DataFrame(distribution_rows).to_csv(
        os.path.join(output_dir, "split_distribution_audit.csv"), index=False
    )
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(output_dir, "split_distribution_pairwise.csv"), index=False
    )
    _plot_split_distributions(task_records, output_dir)
    with open(
        os.path.join(output_dir, "split_assignment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "seed": int(seed),
                "patient_disjoint": True,
                "source_algorithm": (
                    "class-stratified patient candidate search balanced on raw "
                    "value and abnormal-score distributions"
                ),
                "target_results": selections,
            },
            handle,
            indent=2,
        )
    with open(
        os.path.join(output_dir, "dataset_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "source_dir": os.path.abspath(source_dir),
                "raw_root": os.path.abspath(raw_root),
                "source_timezone": source_quality["lab_report_time"][
                    "source_timezone"
                ],
                "lab_video_max_delta_hours": 24.0,
                "video_input": {
                    "height": VIDEO_HEIGHT,
                    "width": VIDEO_WIDTH,
                    "channels": 3,
                    "interpolation": False,
                    "frame_sampling": False,
                },
                "ecg_input": {
                    "resampling": "linear interpolation on source Unix timestamps",
                    "sample_rate_hz": ECG_SAMPLE_RATE_HZ,
                    "samples_per_window": ECG_SAMPLES_PER_WINDOW,
                    "extrapolation": False,
                    "input_channels": ["robust_normalized_amplitude"],
                    "max_source_gap_seconds": (
                        ECG_MAX_INTERPOLATION_GAP_SECONDS
                    ),
                },
                "window_seconds": WINDOW_SECONDS,
                "window_stride_seconds": WINDOW_STRIDE_SECONDS,
                "candidate_videos": int(len(quality)),
                "valid_recordings": int(len(recordings)),
                "base_windows": int(len(base_windows)),
                "targets": list(task_records),
            },
            handle,
            indent=2,
        )
    all_windows = pd.concat(task_window_frames, ignore_index=True)
    return task_records, all_windows, summary_frame, recordings, quality


class VideoEcgWindowDataset(Dataset):
    def __init__(self, windows, training=False):
        self.windows = windows.reset_index(drop=True).copy()
        self.training = bool(training)
        self._ecg_cache = OrderedDict()
        self._timestamp_cache = OrderedDict()

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ecg_cache"] = OrderedDict()
        state["_timestamp_cache"] = OrderedDict()
        return state

    @staticmethod
    def _cached(cache, key, limit, loader):
        value = cache.pop(key, None)
        if value is None:
            value = loader(key)
        cache[key] = value
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _video_times(self, path):
        return self._cached(
            self._timestamp_cache,
            path,
            TIMESTAMP_CACHE_RECORDINGS,
            _read_video_times,
        )

    def _ecg(self, path):
        return self._cached(
            self._ecg_cache,
            path,
            ECG_CACHE_RECORDINGS,
            _read_ecg,
        )

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
        frame_start, frame_end = int(row.frame_start_index), int(row.frame_end_index)
        frames = self._decode_frames(row.video_path, frame_start, frame_end)
        video_times = self._video_times(row.video_times_path)[frame_start:frame_end]
        if len(video_times) != len(frames):
            raise RuntimeError(f"Frame/timestamp mismatch for {row.window_id}")

        ecg_times, ecg_signal = self._ecg(row.ecg_path)
        ecg_start, ecg_end = int(row.ecg_start_index), int(row.ecg_end_index)
        source_times = ecg_times[ecg_start:ecg_end]
        source_signal = ecg_signal[ecg_start:ecg_end]
        if len(source_times) != len(source_signal):
            raise RuntimeError(f"ECG timestamp/signal mismatch for {row.window_id}")
        start = float(row.window_start_unix)
        uniform_times = start + (
            np.arange(ECG_SAMPLES_PER_WINDOW, dtype=np.float64)
            / ECG_SAMPLE_RATE_HZ
        )
        if (
            len(source_times) < 2
            or source_times[0] > uniform_times[0]
            or source_times[-1] < uniform_times[-1]
        ):
            raise RuntimeError(f"ECG resampling would extrapolate for {row.window_id}")
        max_gap = float(np.max(np.diff(source_times)))
        if max_gap > ECG_MAX_INTERPOLATION_GAP_SECONDS + 1e-9:
            raise RuntimeError(
                f"ECG source gap {max_gap:.6f}s exceeds limit for {row.window_id}"
            )
        uniform_signal = np.interp(
            uniform_times,
            source_times,
            source_signal.astype(np.float64, copy=False),
        ).astype(np.float32)
        median = float(np.median(uniform_signal))
        mad = float(np.median(np.abs(uniform_signal - median))) * 1.4826
        scale = mad if mad > 1e-6 else max(float(np.std(uniform_signal)), 1.0)
        uniform_signal = np.clip(
            (uniform_signal - median) / scale, -8.0, 8.0
        )
        ecg = torch.from_numpy(uniform_signal[:, None])

        frame_times = torch.from_numpy((video_times - start).astype(np.float32))
        ecg_relative_times = torch.from_numpy(
            (uniform_times - start).astype(np.float32)
        )
        if self.training and torch.rand(()) < HORIZONTAL_FLIP_PROBABILITY:
            frames = torch.flip(frames, dims=(-1,))
        return {
            "frames": frames,
            "frame_times": frame_times,
            "ecg": ecg,
            "ecg_times": ecg_relative_times,
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
        "ecg": pad_sequence(
            [sample["ecg"] for sample in samples],
            batch_first=True,
            padding_value=0.0,
        ),
        "ecg_times": pad_sequence(
            [sample["ecg_times"] for sample in samples],
            batch_first=True,
            padding_value=0.0,
        ),
        "ecg_lengths": torch.tensor(
            [len(sample["ecg"]) for sample in samples], dtype=torch.long
        ),
        "targets": torch.stack([sample["target_score"] for sample in samples]),
        "window_ids": [sample["window_id"] for sample in samples],
        "video_ids": [sample["video_id"] for sample in samples],
        "hospital_ids": [sample["hospital_id"] for sample in samples],
    }


__all__ = [
    "VideoEcgWindowDataset",
    "build_recording_index",
    "collate_windows",
    "prepare_experiment_data",
    "validate_lab_time_alignment",
]
