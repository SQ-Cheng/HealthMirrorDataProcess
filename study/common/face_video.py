"""Resolve validated 224 crops and decode independently coded FFV1 packets."""

from collections import OrderedDict
import json
import mmap
import os
import fcntl
from pathlib import Path
from functools import lru_cache

import av
import numpy as np
import pandas as pd
import torch
from torchvision.io import ImageReadMode, decode_jpeg
from .time_alignment import read_session_metadata


def raw_root():
    return Path(os.environ.get("HEALTHMIRROR_RAW_ROOT", "/root/shared/HealthMirrorRawData"))


RAW_SOURCE_ERRORS = (
    "JPEG/timestamp count mismatch:", "Too few raw frames:",
    "Non-contiguous timestamp frame IDs:", "Invalid or nonmonotonic recorder timestamps:",
    "Unexpected timestamp schema:",
)


@lru_cache(maxsize=1)
def _processing_rejections(path, mtime_ns, size):
    table = pd.read_csv(path)
    if "error" not in table:
        return {}
    failed = table.loc[table.status.eq("failed")]
    return {str(row.video_id): str(row.error) for row in failed.itertuples()
            if str(row.error).startswith(RAW_SOURCE_ERRORS)}


def source_rejection(path):
    ledger = raw_root() / "_face224_processing/index.csv"
    if not ledger.is_file():
        return None
    with (ledger.parent / ".processing.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        stat = ledger.stat()
        rejections = _processing_rejections(str(ledger), stat.st_mtime_ns, stat.st_size)
    path = Path(path)
    video_id = f"{path.parent.parent.name.removesuffix('_data')}_{path.parent.name}"
    return rejections.get(video_id)


def face_source_mode():
    mode = os.environ.get("HEALTHMIRROR_FACE_SOURCE", "auto")
    if mode not in {"auto", "face224", "legacy128"}:
        raise ValueError(f"Invalid HEALTHMIRROR_FACE_SOURCE: {mode}")
    if mode == "auto":
        mode = "face224" if (raw_root() / "_face224_processing/protocol.json").is_file() else "legacy128"
    return mode


def resolve_video(row, legacy_root):
    directory = f"{row.mirror}_data/patient_{int(row.lab_patient_id):06d}"
    if face_source_mode() == "legacy128":
        return str(Path(legacy_root) / directory / "video.avi")
    path = raw_root() / directory / "face224.mkv"
    source = path.with_name("raw_video.avi")
    if not source.is_file() or not source.stat().st_size:
        if path.exists():
            raise RuntimeError(f"Stale 224 result without a usable raw source: {path}")
        return str(path)
    metadata_path = path.with_name("face224_metadata.json")
    if not metadata_path.is_file():
        if not path.exists() and source_rejection(path):
            return str(path)
        raise RuntimeError(f"224 preprocessing is incomplete or failed: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    protocol = metadata["protocol"]
    if (protocol["quality"]["confidence_threshold"] != .75
            or protocol["geometry"].get("enable_alignment", True)
            or protocol["size"] != [224, 224]):
        raise RuntimeError(f"224 video has the wrong crop protocol: {path}")
    if metadata["video_id"] != str(row.video_id):
        raise RuntimeError(f"224 session identity mismatch: {path}")
    if metadata["status"] == "no_valid_frames":
        return str(path)
    if hasattr(row, "hospital_id"):
        session = read_session_metadata(path.parent / "patient_info.txt",
                                        expected_local_id=row.lab_patient_id,
                                        expected_hospital_id=row.hospital_id)
        if metadata.get("session_time_unix") is None or abs(session["session_time_unix"] - metadata["session_time_unix"]) > 1e-6:
            raise RuntimeError(f"224 canonical session timestamp mismatch: {path}")
    if metadata["status"] != "completed" or not metadata.get("verification", {}).get("pixel_exact"):
        raise RuntimeError(f"Unverified 224 video: {path}")
    if not path.is_file() or path.stat().st_size != metadata["bytes"]:
        raise RuntimeError(f"224 video is missing or changed: {path}")
    for source, signature in metadata["source_signatures"].items():
        stat = Path(source).stat()
        if stat.st_size != signature["size"] or stat.st_mtime_ns != signature["mtime_ns"]:
            raise RuntimeError(f"Raw source changed since cropping: {source}")
    return str(path)


def scan_ffv1(path, frame_policy, quantiles, minimum_gap):
    """Locate exact packet payload offsets, without persisting a second cache."""
    path = Path(path)
    if not path.with_name("face224_metadata.json").exists():
        rejection = source_rejection(path)
        if rejection:
            return [], [], [], "", [{"source_frame_index": -1, "byte_start": -1,
                                    "reason": f"invalid_original_frame_timestamp_mapping: {rejection}"}]
        source = path.with_name("raw_video.avi")
        if not source.is_file() or not source.stat().st_size:
            return [], [], [], "", [{"source_frame_index": -1, "byte_start": -1,
                                    "reason": "missing_or_empty_original_rgb_video"}]
        raise RuntimeError(f"Incomplete 224 preprocessing: {path}")
    metadata = json.loads(path.with_name("face224_metadata.json").read_text())
    if metadata["status"] == "no_valid_frames":
        return [], [], [], "", [{"source_frame_index": -1, "byte_start": -1, "reason": "no_valid_face224_frames"}]
    mapping = pd.read_csv(path.with_name("face224_frames.csv"))
    indices = mapping.source_frame_index.to_numpy(np.int64)
    if (len(mapping) != metadata["retained_frames"]
            or not np.array_equal(mapping.output_frame_index, np.arange(len(mapping)))
            or np.any(np.diff(indices) <= 0)
            or np.any(np.diff(mapping.source_elapsed_seconds) <= 0)
            or np.any(np.diff(mapping.encoded_pts) <= 0)):
        raise ValueError(f"Invalid retained-frame mapping: {path}")
    if "confidence" in mapping and (mapping.confidence.isna().any() or mapping.confidence.lt(.75).any()):
        raise ValueError(f"224 frame confidence below acceptance threshold: {path}")
    if metadata.get("session_time_unix") is not None:
        expected = metadata["session_time_unix"] + mapping.source_elapsed_seconds
        if not np.allclose(mapping.canonical_session_frame_time_unix, expected, rtol=0, atol=1e-6):
            raise ValueError(f"224 canonical frame times disagree with session: {path}")
    if frame_policy == "20frame":
        selected = np.rint(np.asarray(quantiles) * (len(mapping) - 1)).astype(int)
        if (len(set(selected)) != len(quantiles) or np.min(np.diff(indices[selected])) < minimum_gap):
            selected = np.rint(np.linspace(0, len(mapping) - 1, len(quantiles))).astype(int)
        if (len(set(selected)) != len(quantiles) or np.min(np.diff(indices[selected])) < minimum_gap):
            return [], [], [], "", [{"source_frame_index": -1, "byte_start": -1,
                                    "reason": "cannot_select_20_nonadjacent_accepted_frames"}]
    else:
        selected = np.arange(len(mapping))
    starts, ends, source_indices = [], [], []
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped, av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != "ffv1" or (stream.width, stream.height) != (224, 224):
            raise ValueError(f"Expected FFV1 224x224 source: {path}")
        extra = (stream.codec_context.extradata or b"").hex()
        def record_packet(packet, position):
            if not packet.is_keyframe:
                raise ValueError(f"FFV1 packet is not independently coded: {path}")
            if packet.pts != int(mapping.encoded_pts.iloc[position]):
                raise ValueError(f"FFV1 packet/sidecar timestamp mismatch: {path}")
            if str(packet.time_base) != str(mapping.encoded_time_base.iloc[position]):
                raise ValueError(f"FFV1 time base changed: {path}")
            payload = bytes(packet)
            # Matroska packet.pos points at the block header, not necessarily
            # the payload. Locate and verify the exact encoded bytes nearby.
            if packet.pos is None or packet.pos < 0:
                raise ValueError(f"Missing FFV1 packet position: {path}")
            start = mapped.find(payload, packet.pos, packet.pos + packet.size + 256)
            if start < 0:
                raise ValueError(f"Cannot locate FFV1 packet payload: {path}, {position}")
            starts.append(start)
            ends.append(start + packet.size)
            source_indices.append(int(indices[position]))

        if frame_policy == "20frame":
            # GOP 1 and Matroska cues permit direct access to the selected PTS;
            # do not read every large FFV1 payload merely to select twenty.
            for position in selected:
                target_pts = int(mapping.encoded_pts.iloc[position])
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                for packet in container.demux(stream):
                    if not packet.size or packet.pts is None or packet.pts < target_pts:
                        continue
                    record_packet(packet, position)
                    break
                else:
                    raise ValueError(f"Selected FFV1 frame is missing: {path}, {position}")
        else:
            count = 0
            for packet in container.demux(stream):
                if not packet.size:
                    continue
                if count >= len(mapping):
                    raise ValueError(f"FFV1 packet count mismatch: {path}")
                record_packet(packet, count)
                count += 1
            if count != len(mapping):
                raise ValueError(f"Incomplete FFV1 packet stream: {path}")
    if len(starts) != len(selected):
        raise ValueError(f"Incomplete FFV1 packet index: {path}")
    return starts, ends, source_indices, extra, []


def decode_indexed_frame(index, video_position, global_index, handle, decoders):
    start, end = int(index.starts[global_index]), int(index.ends[global_index])
    handle.seek(start)
    payload = handle.read(end - start)
    if len(payload) != end - start:
        raise RuntimeError("Truncated indexed frame payload")
    if str(index.video_formats[video_position]) == "ffv1":
        decoder = decoders.pop(video_position, None)
        if decoder is None:
            decoder = av.CodecContext.create("ffv1", "r")
            decoder.extradata = bytes.fromhex(str(index.codec_extradata[video_position]))
            decoder.width = decoder.height = 224
            decoder.pix_fmt = "bgr0"
            decoder.thread_count = 1
        decoders[video_position] = decoder
        while len(decoders) > 16:
            decoders.popitem(last=False)
        frames = decoder.decode(av.Packet(payload))
        if len(frames) != 1:
            raise RuntimeError("FFV1 packet did not decode to one independent frame")
        image = torch.from_numpy(frames[0].to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous()
        expected_size = 224
    else:
        image = decode_jpeg(torch.frombuffer(bytearray(payload), dtype=torch.uint8), mode=ImageReadMode.RGB, device="cpu")
        expected_size = 128
    if tuple(image.shape) != (3, expected_size, expected_size):
        raise RuntimeError(f"Unexpected indexed frame shape: {tuple(image.shape)}")
    return image
