"""Compact payload indexes for native 224 FFV1 or legacy 128 MJPEG frames."""

from dataclasses import dataclass
from io import BytesIO
import json
import mmap
import os

import numpy as np
import pandas as pd
from PIL import Image

from study.common.face_video import face_source_mode, resolve_video, scan_ffv1

from .config import (
    DATA_ROOT,
    FRAME_QUANTILES,
    FRAMES_PER_VIDEO,
    MIN_SOURCE_FRAME_GAP,
)


INDEX_SCHEMA_VERSIONS = {"20frame": 4, "allframes": 5}


def video_path_for_row(row):
    return resolve_video(row, DATA_ROOT)


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
                    if size != (128, 128):
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
                    if size != (128, 128):
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


def _index_is_reusable(index_dir, expected_video_ids, frame_policy, expected_paths=None):
    index_path = os.path.join(index_dir, "frame_offsets.npz")
    manifest_path = os.path.join(index_dir, "index_manifest.json")
    if not os.path.exists(index_path) or not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        version = manifest.get("schema_version")
        legacy_version = {"20frame": 2, "allframes": 3}[frame_policy]
        if version != INDEX_SCHEMA_VERSIONS[frame_policy] and not (
            face_source_mode() == "legacy128" and version == legacy_version
        ):
            return False
        if manifest.get("face_source", "legacy128") != face_source_mode():
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
            for path, signature in row.get("sidecars", {}).items():
                sidecar_stat = os.stat(path)
                if [sidecar_stat.st_size, sidecar_stat.st_mtime_ns] != signature:
                    return False
            if expected_paths and row["video_id"] in expected_paths and row["video_path"] != expected_paths[row["video_id"]]:
                return False
            stat = os.stat(row.get("fingerprint_path", row["video_path"]))
            if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                return False
        with np.load(index_path, allow_pickle=False) as index:
            required = {"video_ids", "video_paths", "video_ptr", "starts", "ends", "source_indices"}
            if version == INDEX_SCHEMA_VERSIONS[frame_policy]:
                required.update({"video_formats", "codec_extradata"})
            if not required.issubset(index.files):
                return False
            if int(index["video_ptr"][-1]) != len(index["starts"]):
                return False
            if "video_formats" in index.files:
                codec = "ffv1" if face_source_mode() == "face224" else "mjpeg"
                if len(index["video_formats"]) != len(index["video_ids"]) or not np.all(index["video_formats"] == codec):
                    return False
    except Exception:
        return False
    return True


def build_or_reuse_frame_index(video_records, index_dir, frame_policy="20frame"):
    """Index independent encoded payloads without persisting decoded pixels."""
    if frame_policy not in INDEX_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported frame policy: {frame_policy}")
    os.makedirs(index_dir, exist_ok=True)
    if "hospital_id" in video_records and video_records.groupby("video_id").hospital_id.nunique().gt(1).any():
        raise ValueError("Source video is associated with multiple hospital IDs")
    videos = video_records[
        ["video_id", "mirror", "lab_patient_id"] + (["hospital_id"] if "hospital_id" in video_records else [])
    ].drop_duplicates("video_id").sort_values("video_id").reset_index(drop=True)
    expected_video_ids = videos["video_id"].astype(str).tolist()
    expected_paths = {str(row.video_id): video_path_for_row(row) for row in videos.itertuples(index=False)}
    index_path = os.path.join(index_dir, "frame_offsets.npz")
    if _index_is_reusable(index_dir, expected_video_ids, frame_policy, expected_paths):
        print(f"Reusing compact {frame_policy} index: {index_path}", flush=True)
        return FrameOffsetIndex.load(index_path)

    all_starts, all_ends, all_source_indices = [], [], []
    video_ids, video_paths, video_ptr = [], [], [0]
    video_formats, codec_extradata = [], []
    summary_rows, failure_rows, manifest_rows, failed_manifest_rows = [], [], [], []
    print(
        f"Building compact {frame_policy} {face_source_mode()} index for {len(videos)} videos",
        flush=True,
    )
    for position, row in enumerate(videos.itertuples(index=False), start=1):
        video_path = expected_paths[str(row.video_id)]
        is_ffv1 = video_path.endswith(".mkv")
        fingerprint_path = video_path if os.path.isfile(video_path) else os.path.join(os.path.dirname(video_path), "face224_metadata.json")
        if is_ffv1 and not os.path.isfile(fingerprint_path):
            fingerprint_path = os.path.dirname(video_path)
            while not os.path.exists(fingerprint_path):
                fingerprint_path = os.path.dirname(fingerprint_path)
        if not os.path.isfile(video_path) and not is_ffv1:
            raise FileNotFoundError(f"Missing source video: {video_path}")
        extra = ""
        if is_ffv1:
            starts, ends, source_indices, extra, failures = scan_ffv1(video_path, frame_policy, FRAME_QUANTILES, MIN_SOURCE_FRAME_GAP)
        elif frame_policy == "20frame":
            starts, ends, source_indices, failures = _scan_20_frames(video_path)
        else:
            starts, ends, source_indices, failures = _scan_all_frames(video_path)
        if frame_policy == "20frame":
            valid_video = len(starts) == FRAMES_PER_VIDEO
            excluded_status = "excluded_cannot_select_20_frames"
            excluded_reason = "cannot_select_20_nonadjacent_frames"
        else:
            valid_video = bool(starts)
            excluded_status = "excluded_no_decodable_frames"
            excluded_reason = "no_decodable_frames"
        stat = os.stat(fingerprint_path)
        sidecars = {}
        if is_ffv1:
            for name in ("face224_frames.csv", "face224_metadata.json",
                         "raw_video.avi", "video.avi.ts", "patient_info.txt"):
                path = os.path.join(os.path.dirname(video_path), name)
                if os.path.isfile(path):
                    sidecar_stat = os.stat(path)
                    sidecars[path] = [sidecar_stat.st_size, sidecar_stat.st_mtime_ns]
        if not valid_video:
            if failures:
                excluded_reason = failures[-1]["reason"]
            summary_rows.append({
                "video_id": str(row.video_id),
                "video_path": video_path,
                "valid_frames": 0,
                "invalid_frames": len(failures),
                "size_bytes": stat.st_size,
                "status": excluded_status,
                "reason": excluded_reason,
            })
            failed_manifest_rows.append({
                "video_id": str(row.video_id),
                "video_path": video_path,
                "fingerprint_path": fingerprint_path,
                "sidecars": sidecars,
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
        video_formats.append("ffv1" if is_ffv1 else "mjpeg")
        codec_extradata.append(extra)
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
            "sidecars": sidecars,
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
        video_formats=np.asarray(video_formats, dtype=str),
        codec_extradata=np.asarray(codec_extradata, dtype=str),
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
            "storage_policy": "independent encoded frame/packet byte offsets only; no decoded frames persisted",
            "face_source": face_source_mode(),
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
    video_formats: np.ndarray = None
    codec_extradata: np.ndarray = None

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            names = (
                "video_ids", "video_paths", "video_ptr", "starts", "ends", "source_indices"
            )
            kwargs = {name: values[name] for name in names}
            kwargs.update({name: values[name] for name in ("video_formats", "codec_extradata") if name in values.files})
            return cls(**kwargs)

    def __post_init__(self):
        if self.video_formats is None:
            self.video_formats = np.full(len(self.video_ids), "mjpeg")
        if self.codec_extradata is None:
            self.codec_extradata = np.full(len(self.video_ids), "")
        self.video_lookup = {
            str(video_id): index for index, video_id in enumerate(self.video_ids)
        }

    def frame_range(self, video_id):
        video_index = self.video_lookup[str(video_id)]
        return int(self.video_ptr[video_index]), int(self.video_ptr[video_index + 1])
