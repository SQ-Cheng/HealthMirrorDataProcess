"""Compact byte-offset index for 20 nonadjacent MJPEG face frames per video."""

from io import BytesIO
import json
import mmap
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    CACHE_DIR,
    FRAME_QUANTILES,
    FRAMES_PER_VIDEO,
    MIN_SOURCE_FRAME_GAP,
    SOURCE_IMAGE_SIZE,
)


SCHEMA_VERSION = 1


class FrameOffsetIndex:
    def __init__(self, arrays):
        for name, value in arrays.items():
            setattr(self, name, value)
        self.video_lookup = {
            str(video_id): index for index, video_id in enumerate(self.video_ids)
        }

    def frame_range(self, video_id):
        position = self.video_lookup[str(video_id)]
        return int(self.video_ptr[position]), int(self.video_ptr[position + 1])

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: archive[name] for name in archive.files})


def _jpeg_ranges(video_path):
    ranges = []
    if video_path.stat().st_size == 0:
        return ranges
    with open(video_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            position = 0
            while True:
                start = mapped.find(b"\xff\xd8", position)
                if start < 0:
                    break
                marker = mapped.find(b"\xff\xd9", start + 2)
                if marker < 0:
                    break
                end = marker + 2
                ranges.append((start, end))
                position = end
    return ranges


def _select_frames(video_path):
    ranges = _jpeg_ranges(video_path)
    if len(ranges) < FRAMES_PER_VIDEO:
        return None, f"too_few_complete_jpeg_frames={len(ranges)}"
    targets = [int(round(q * (len(ranges) - 1))) for q in FRAME_QUANTILES]
    if len(set(targets)) != FRAMES_PER_VIDEO or min(np.diff(targets)) < MIN_SOURCE_FRAME_GAP:
        targets = np.round(np.linspace(0, len(ranges) - 1, FRAMES_PER_VIDEO)).astype(int).tolist()
    if len(set(targets)) != FRAMES_PER_VIDEO or min(np.diff(targets)) < MIN_SOURCE_FRAME_GAP:
        return None, f"cannot_select_{FRAMES_PER_VIDEO}_nonadjacent_frames"
    selected, used, previous = [], set(), -MIN_SOURCE_FRAME_GAP
    with open(video_path, "rb") as handle:
        for target in targets:
            candidate = None
            offsets = [0]
            for distance in range(1, 31):
                offsets.extend((-distance, distance))
            for offset in offsets:
                source_index = target + offset
                if (
                    source_index < 0
                    or source_index >= len(ranges)
                    or source_index in used
                    or source_index - previous < MIN_SOURCE_FRAME_GAP
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
                except Exception:
                    continue
                candidate = (start, end, source_index)
                break
            if candidate is None:
                return None, f"no_decodable_frame_near_source_index={target}"
            selected.append(candidate)
            used.add(candidate[2])
            previous = candidate[2]
    return selected, ""


def _cache_valid(manifest_path, index_path, records):
    if not manifest_path.is_file() or not index_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            return False
        expected = set(records["video_id"].astype(str))
        observed = set(manifest.get("requested_video_ids", []))
        if expected != observed:
            return False
        for row in manifest.get("source_files", []):
            stat = Path(row["path"]).stat()
            if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
                return False
        index = FrameOffsetIndex.load(index_path)
        return int(index.video_ptr[-1]) == len(index.starts)
    except Exception:
        return False


def build_or_reuse_frame_index(records, cache_dir=CACHE_DIR):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "frame_offsets.npz"
    manifest_path = cache_dir / "index_manifest.json"
    if _cache_valid(manifest_path, index_path, records):
        index = FrameOffsetIndex.load(index_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"[frames] reused {len(index.video_ids)}-video index", flush=True)
        return index, manifest

    starts, ends, source_indices = [], [], []
    video_ids, video_paths, video_ptr = [], [], [0]
    source_files, failures = [], []
    for position, row in enumerate(records.itertuples(index=False), start=1):
        path = Path(row.video_path)
        stat = path.stat()
        source_files.append({
            "video_id": str(row.video_id),
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
        selected, reason = _select_frames(path)
        if selected is None:
            failures.append({"video_id": str(row.video_id), "path": str(path), "reason": reason})
            continue
        for start, end, source_index in selected:
            starts.append(start); ends.append(end); source_indices.append(source_index)
        video_ids.append(str(row.video_id)); video_paths.append(str(path)); video_ptr.append(len(starts))
        if position % 100 == 0 or position == len(records):
            print(f"[frames] scanned={position}/{len(records)} usable={len(video_ids)}", flush=True)
    arrays = {
        "video_ids": np.asarray(video_ids, dtype="U64"),
        "video_paths": np.asarray(video_paths, dtype="U512"),
        "video_ptr": np.asarray(video_ptr, dtype=np.int64),
        "starts": np.asarray(starts, dtype=np.int64),
        "ends": np.asarray(ends, dtype=np.int64),
        "source_indices": np.asarray(source_indices, dtype=np.int32),
    }
    np.savez(index_path, **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "frames_per_video": FRAMES_PER_VIDEO,
            "quantile_range": [FRAME_QUANTILES[0], FRAME_QUANTILES[-1]],
            "minimum_source_frame_gap": MIN_SOURCE_FRAME_GAP,
            "source_image_size": SOURCE_IMAGE_SIZE,
        },
        "requested_video_ids": records["video_id"].astype(str).tolist(),
        "usable_video_count": len(video_ids),
        "failed_video_count": len(failures),
        "failures": failures,
        "source_files": source_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not video_ids:
        raise RuntimeError("No recovery videos have 20 decodable nonadjacent frames")
    return FrameOffsetIndex(arrays), manifest
