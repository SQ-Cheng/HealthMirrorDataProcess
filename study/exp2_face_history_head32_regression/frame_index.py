"""Compact byte-offset indexes for selected or all decodable MJPEG frames."""

from dataclasses import dataclass
from io import BytesIO
import json
import mmap
import os

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    DATA_ROOT,
    FRAME_QUANTILES,
    FRAMES_PER_VIDEO,
    MIN_SOURCE_FRAME_GAP,
    SOURCE_IMAGE_SIZE,
)


INDEX_SCHEMA_VERSIONS = {"20frame": 2, "allframes": 3}


def video_path_for_row(row):
    return os.path.join(
        DATA_ROOT,
        f"{row.mirror}_data",
        f"patient_{int(row.lab_patient_id):06d}",
        "video.avi",
    )


def _scan_20_frames(video_path):
    ranges = []
    if os.path.getsize(video_path) == 0:
        return [], [], [], [{
            "source_frame_index": -1,
            "byte_start": -1,
            "reason": "empty_video_file",
        }]
    with open(video_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = 0
            while True:
                start = mapped.find(b"\xff\xd8", position)
                if start < 0:
                    break
                end_marker = mapped.find(b"\xff\xd9", start + 2)
                if end_marker < 0:
                    break
                end = end_marker + 2
                ranges.append((start, end))
                position = end
    if not ranges:
        return [], [], [], [{
            "source_frame_index": -1,
            "byte_start": -1,
            "reason": "no_complete_jpeg_ranges",
        }]

    target_indices = [
        int(round(quantile * (len(ranges) - 1)))
        for quantile in FRAME_QUANTILES
    ]
    if (
        len(target_indices) != FRAMES_PER_VIDEO
        or len(set(target_indices)) != FRAMES_PER_VIDEO
        or min(np.diff(target_indices)) < MIN_SOURCE_FRAME_GAP
    ):
        target_indices = np.round(
            np.linspace(0, len(ranges) - 1, FRAMES_PER_VIDEO)
        ).astype(int).tolist()
    if (
        len(set(target_indices)) != FRAMES_PER_VIDEO
        or min(np.diff(target_indices)) < MIN_SOURCE_FRAME_GAP
    ):
        return [], [], [], [{
            "source_frame_index": -1,
            "byte_start": -1,
            "reason": (
                f"too_few_source_frames={len(ranges)} "
                f"for_{FRAMES_PER_VIDEO}_nonadjacent_frames"
            ),
        }]

    starts, ends, source_indices, failures = [], [], [], []
    used = set()
    previous_index = -MIN_SOURCE_FRAME_GAP
    with open(video_path, "rb") as handle:
        for target_index in target_indices:
            offsets = [0]
            for distance in range(1, 31):
                offsets.extend((-distance, distance))
            selected = None
            for offset in offsets:
                source_index = target_index + offset
                if (
                    source_index < 0
                    or source_index >= len(ranges)
                    or source_index in used
                    or source_index - previous_index < MIN_SOURCE_FRAME_GAP
                ):
                    continue
                start, end = ranges[source_index]
                try:
                    handle.seek(start)
                    payload = handle.read(end - start)
                    with Image.open(BytesIO(payload)) as image:
                        size = image.size
                        image.verify()
                    if size != (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
                        raise ValueError(f"unexpected_frame_size={size}")
                except Exception as exc:
                    failures.append({
                        "source_frame_index": source_index,
                        "byte_start": start,
                        "reason": str(exc),
                    })
                    continue
                selected = (source_index, start, end)
                break
            if selected is None:
                failures.append({
                    "source_frame_index": target_index,
                    "byte_start": ranges[target_index][0],
                    "reason": "no_decodable_nonadjacent_frame_within_search_radius",
                })
                return [], [], [], failures
            source_index, start, end = selected
            used.add(source_index)
            previous_index = source_index
            source_indices.append(source_index)
            starts.append(start)
            ends.append(end)
    return starts, ends, source_indices, failures


def _scan_all_frames(video_path):
    starts, ends, source_indices, failures = [], [], [], []
    if os.path.getsize(video_path) == 0:
        return [], [], [], [{
            "source_frame_index": -1,
            "byte_start": -1,
            "reason": "empty_video_file",
        }]
    with open(video_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = 0
            source_index = 0
            while True:
                start = mapped.find(b"\xff\xd8", position)
                if start < 0:
                    break
                end_marker = mapped.find(b"\xff\xd9", start + 2)
                if end_marker < 0:
                    failures.append({
                        "source_frame_index": source_index,
                        "byte_start": start,
                        "reason": "missing_jpeg_end_marker",
                    })
                    break
                end = end_marker + 2
                try:
                    payload = mapped[start:end]
                    with Image.open(BytesIO(payload)) as image:
                        size = image.size
                        image.verify()
                    if size != (SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
                        raise ValueError(f"unexpected_frame_size={size}")
                    starts.append(start)
                    ends.append(end)
                    source_indices.append(source_index)
                except Exception as exc:
                    failures.append({
                        "source_frame_index": source_index,
                        "byte_start": start,
                        "reason": str(exc),
                    })
                position = end
                source_index += 1
    if not starts and not failures:
        failures.append({
            "source_frame_index": -1,
            "byte_start": -1,
            "reason": "no_complete_jpeg_ranges",
        })
    return starts, ends, source_indices, failures


def _index_is_reusable(index_dir, expected_video_ids, frame_policy):
    index_path = os.path.join(index_dir, "frame_offsets.npz")
    manifest_path = os.path.join(index_dir, "index_manifest.json")
    if not os.path.exists(index_path) or not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != INDEX_SCHEMA_VERSIONS[frame_policy]:
            return False
        policy = manifest.get("frame_policy", {})
        if frame_policy == "20frame":
            if policy.get("frames_per_video") != FRAMES_PER_VIDEO:
                return False
        elif policy.get("mode") != "all_decodable_frames":
            return False
        rows = manifest.get("videos", [])
        failed_rows = manifest.get("failed_videos", [])
        indexed_or_failed = {
            row["video_id"] for row in (*rows, *failed_rows)
        }
        if not set(expected_video_ids).issubset(indexed_or_failed):
            return False
        for row in (*rows, *failed_rows):
            stat = os.stat(row["video_path"])
            if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                return False
        with np.load(index_path, allow_pickle=False) as index:
            required = {"video_ids", "video_paths", "video_ptr", "starts", "ends", "source_indices"}
            if not required.issubset(index.files):
                return False
            if int(index["video_ptr"][-1]) != len(index["starts"]):
                return False
    except Exception:
        return False
    return True


def build_or_reuse_frame_index(video_records, index_dir, frame_policy="20frame"):
    """Index JPEG byte ranges without persisting decoded frame pixels."""
    if frame_policy not in INDEX_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported frame policy: {frame_policy}")
    os.makedirs(index_dir, exist_ok=True)
    videos = video_records[
        ["video_id", "mirror", "lab_patient_id"]
    ].drop_duplicates("video_id").sort_values("video_id").reset_index(drop=True)
    expected_video_ids = videos["video_id"].astype(str).tolist()
    index_path = os.path.join(index_dir, "frame_offsets.npz")
    if _index_is_reusable(index_dir, expected_video_ids, frame_policy):
        print(f"Reusing compact {frame_policy} index: {index_path}", flush=True)
        return FrameOffsetIndex.load(index_path)

    all_starts, all_ends, all_source_indices = [], [], []
    video_ids, video_paths, video_ptr = [], [], [0]
    summary_rows, failure_rows, manifest_rows, failed_manifest_rows = [], [], [], []
    print(
        f"Building compact {frame_policy} MJPEG index for {len(videos)} videos",
        flush=True,
    )
    for position, row in enumerate(videos.itertuples(index=False), start=1):
        video_path = video_path_for_row(row)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Missing source video: {video_path}")
        if frame_policy == "20frame":
            starts, ends, source_indices, failures = _scan_20_frames(video_path)
            valid_video = len(starts) == FRAMES_PER_VIDEO
            excluded_status = "excluded_cannot_select_20_frames"
            excluded_reason = "cannot_select_20_nonadjacent_frames"
        else:
            starts, ends, source_indices, failures = _scan_all_frames(video_path)
            valid_video = bool(starts)
            excluded_status = "excluded_no_decodable_frames"
            excluded_reason = "no_decodable_frames"
        stat = os.stat(video_path)
        if not valid_video:
            summary_rows.append({
                "video_id": str(row.video_id),
                "video_path": video_path,
                "valid_frames": 0,
                "invalid_frames": len(failures),
                "size_bytes": stat.st_size,
                "status": excluded_status,
            })
            failed_manifest_rows.append({
                "video_id": str(row.video_id),
                "video_path": video_path,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "reason": excluded_reason,
            })
            for failure in failures:
                failure_rows.append({"video_id": str(row.video_id), **failure})
            print(f"  excluded {row.video_id}: {excluded_reason}", flush=True)
            continue
        video_ids.append(str(row.video_id))
        video_paths.append(video_path)
        all_starts.extend(starts)
        all_ends.extend(ends)
        all_source_indices.extend(source_indices)
        video_ptr.append(len(all_starts))
        summary_rows.append({
            "video_id": str(row.video_id),
            "video_path": video_path,
            "valid_frames": len(starts),
            "invalid_frames": len(failures),
            "size_bytes": stat.st_size,
            "status": "indexed",
        })
        manifest_rows.append({
            "video_id": str(row.video_id),
            "video_path": video_path,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "valid_frames": len(starts),
            "invalid_frames": len(failures),
        })
        for failure in failures:
            failure_rows.append({"video_id": str(row.video_id), **failure})
        if position % 50 == 0 or position == len(videos):
            print(
                f"  indexed {position}/{len(videos)} videos; "
                f"valid_frames={len(all_starts)} invalid_frames={len(failure_rows)}",
                flush=True,
            )

    temporary_path = index_path + ".tmp.npz"
    np.savez_compressed(
        temporary_path,
        video_ids=np.asarray(video_ids, dtype=str),
        video_paths=np.asarray(video_paths, dtype=str),
        video_ptr=np.asarray(video_ptr, dtype=np.int64),
        starts=np.asarray(all_starts, dtype=np.int64),
        ends=np.asarray(all_ends, dtype=np.int64),
        source_indices=np.asarray(all_source_indices, dtype=np.int32),
    )
    os.replace(temporary_path, index_path)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(index_dir, "video_frame_summary.csv"), index=False
    )
    failure_columns = ["video_id", "source_frame_index", "byte_start", "reason"]
    pd.DataFrame(failure_rows, columns=failure_columns).to_csv(
        os.path.join(index_dir, "invalid_frames.csv"), index=False
    )
    with open(
        os.path.join(index_dir, "index_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "schema_version": INDEX_SCHEMA_VERSIONS[frame_policy],
            "storage_policy": "JPEG byte offsets only; no decoded frames persisted",
            "frame_policy": (
                {
                    "mode": "deterministic_nonadjacent_selection",
                    "frames_per_video": FRAMES_PER_VIDEO,
                    "quantiles": list(FRAME_QUANTILES),
                    "minimum_source_frame_gap": MIN_SOURCE_FRAME_GAP,
                }
                if frame_policy == "20frame"
                else {"mode": "all_decodable_frames"}
            ),
            "total_valid_frames": len(all_starts),
            "total_invalid_frames": len(failure_rows),
            "indexed_video_count": len(manifest_rows),
            "failed_video_count": len(failed_manifest_rows),
            "videos": manifest_rows,
            "failed_videos": failed_manifest_rows,
        }, handle, indent=2)
    print(
        f"Saved compact index: frames={len(all_starts)} "
        f"size_bytes={os.path.getsize(index_path)} path={index_path}",
        flush=True,
    )
    return FrameOffsetIndex.load(index_path)


@dataclass
class FrameOffsetIndex:
    video_ids: np.ndarray
    video_paths: np.ndarray
    video_ptr: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    source_indices: np.ndarray

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            return cls(**{name: values[name] for name in (
                "video_ids", "video_paths", "video_ptr", "starts", "ends", "source_indices"
            )})

    def __post_init__(self):
        self.video_lookup = {
            str(video_id): index for index, video_id in enumerate(self.video_ids)
        }

    def frame_range(self, video_id):
        video_index = self.video_lookup[str(video_id)]
        return int(self.video_ptr[video_index]), int(self.video_ptr[video_index + 1])
