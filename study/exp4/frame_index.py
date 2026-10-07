"""Native224 FFV1 packet index; clinical times always use the original recording."""

import json
from pathlib import Path

import numpy as np

from study.common.face_video import face_source_mode
from study.common.time_alignment import read_session_metadata
from study.exp2_face_pretrained_head32_regression.frame_index import (
    FrameOffsetIndex, build_or_reuse_frame_index as shared_build,
)

from .config import CACHE_DIR, FRAMES_PER_VIDEO, TIMEZONE


def build_or_reuse_frame_index(records, cache_dir=CACHE_DIR):
    if face_source_mode() != "face224":
        raise ValueError("Exp4 requires native224 FFV1 crops, not legacy128")
    index = shared_build(records, str(cache_dir), "20frame")
    if set(index.video_formats) != {"ffv1"} or not np.all(np.diff(index.video_ptr) == FRAMES_PER_VIDEO):
        raise RuntimeError("Recovery index contains the wrong codec or frame count")
    lookup = records.set_index("video_id", verify_integrity=True)
    for position, video in enumerate(index.video_ids):
        if str(video) not in lookup.index:
            continue
        row = lookup.loc[str(video)]
        session = read_session_metadata(
            Path(str(index.video_paths[position])).parent / "patient_info.txt",
            expected_local_id=row.lab_patient_id, expected_hospital_id=row.hospital_id, timezone=TIMEZONE,
        )
        if abs(session["session_time_unix"] - float(row.capture_start_unix)) > 1e-6:
            raise RuntimeError(f"Face-source and recovery-label session timestamps differ: {video}")
    manifest = json.loads((Path(cache_dir) / "index_manifest.json").read_text())
    return index, {
        "policy": {**manifest["frame_policy"], "source_image_size": 224, "codec": "FFV1",
                   "storage": "packet offsets only; no decoded-frame cache"},
        "indexed_video_count": manifest["indexed_video_count"],
        "failed_video_count": manifest["failed_video_count"],
    }
