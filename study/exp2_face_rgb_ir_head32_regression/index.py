"""Compact independent FFV1 packet indexes for the selected synchronous frames."""

import json
import mmap
from pathlib import Path

import av
import numpy as np
import pandas as pd

from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from preprocess.face_crop_comparison.run_comparison import sha256
from .config import CACHE, PAIRED


def signature(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def scan_selected(path, mapping, selected):
    starts, ends = [], []
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped, av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != "ffv1" or (stream.width, stream.height) != (224, 224):
            raise ValueError("Expected native224 FFV1")
        extradata = (stream.codec_context.extradata or b"").hex()
        for output_index in selected:
            row = mapping.iloc[int(output_index)]
            target = int(row.encoded_pts)
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for packet in container.demux(stream):
                if not packet.size or packet.pts is None or packet.pts < target:
                    continue
                if packet.pts != target or not packet.is_keyframe or str(packet.time_base) != row.encoded_time_base:
                    raise ValueError("Pair selection/FFV1 packet mismatch")
                payload = bytes(packet)
                start = mapped.find(payload, packet.pos, packet.pos+packet.size+256)
                if start < 0:
                    raise ValueError("Cannot locate independently coded FFV1 payload")
                starts.append(start); ends.append(start+packet.size)
                break
            else:
                raise ValueError("Selected frame is missing")
    return starts, ends, extradata


def build_indexes(video_ids):
    ids = sorted(set(video_ids))
    table = pd.read_csv(PAIRED / "tables/video_pair_counts.csv").set_index("video_id")
    selected = pd.read_csv(PAIRED / "tables/selected20_pairs.csv", float_precision="round_trip")
    groups = {video: group.sort_values("sample_pair_index") for video, group in selected.groupby("video_id")}
    sources = {}
    for video in ids:
        if not table.loc[video, "pairs20_eligible"] or len(groups[video]) != 20:
            raise ValueError("Missing verified paired20 video")
        for mode in ("rgb", "ir"):
            path = Path(table.loc[video, f"{mode}_video_path"])
            prefix = "face224" if mode == "rgb" else "face224_ir"
            for source in (path, path.with_name(prefix+"_frames.csv"), path.with_name(prefix+"_metadata.json")):
                sources[str(source)] = signature(source)
    contract = {"schema_version": 1, "video_ids": ids, "source_signatures": sources,
                "selection_sha256": sha256(PAIRED / "tables/selected20_pairs.csv"),
                "paired_summary_sha256": sha256(PAIRED / "summary.json")}
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "manifest.json"
    if (manifest_path.exists() and json.loads(manifest_path.read_text()) == contract
            and all((CACHE / f"{mode}.npz").exists() for mode in ("rgb", "ir"))):
        print(f"[index-reuse] {len(ids)} paired videos", flush=True)
        return CACHE / "rgb.npz", CACHE / "ir.npz"
    banks = {mode: {"paths": [], "starts": [], "ends": [], "source_indices": [], "extra": []} for mode in ("rgb", "ir")}
    for i, video in enumerate(ids, 1):
        group = groups[video]
        if (group.pair_delta_ms.gt(1000/30).any() or np.any(np.diff(group.rgb_source_frame_index) < 2)
                or np.any(np.diff(group.ir_source_frame_index) < 2)):
            raise ValueError("Pair time/gap policy changed")
        for mode in ("rgb", "ir"):
            path = Path(table.loc[video, f"{mode}_video_path"])
            prefix = "face224" if mode == "rgb" else "face224_ir"
            mapping = pd.read_csv(path.with_name(prefix+"_frames.csv"), float_precision="round_trip")
            positions = group[f"{mode}_output_frame_index"].to_numpy(np.int64)
            if not np.array_equal(mapping.source_frame_index.iloc[positions], group[f"{mode}_source_frame_index"]):
                raise ValueError("Selected original source frames changed")
            starts, ends, extra = scan_selected(path, mapping, positions)
            bank = banks[mode]
            bank["paths"].append(str(path)); bank["starts"].extend(starts); bank["ends"].extend(ends)
            bank["source_indices"].extend(group[f"{mode}_source_frame_index"]); bank["extra"].append(extra)
        if i % 100 == 0:
            print(f"[index-build] {i}/{len(ids)}", flush=True)
    if any(signature(path) != value for path, value in sources.items()):
        raise RuntimeError("Paired video sources changed while indexing")
    for mode, bank in banks.items():
        np.savez_compressed(CACHE / f"{mode}.npz", video_ids=np.asarray(ids), video_paths=np.asarray(bank["paths"]),
            video_ptr=np.arange(len(ids)+1, dtype=np.int64)*20, starts=np.asarray(bank["starts"], dtype=np.int64),
            ends=np.asarray(bank["ends"], dtype=np.int64), source_indices=np.asarray(bank["source_indices"], dtype=np.int64),
            video_formats=np.full(len(ids), "ffv1"), codec_extradata=np.asarray(bank["extra"]))
    manifest_path.write_text(json.dumps(contract, indent=2)+"\n")
    return CACHE / "rgb.npz", CACHE / "ir.npz"


def load_indexes(rgb_path, ir_path):
    rgb, ir = FrameOffsetIndex.load(rgb_path), FrameOffsetIndex.load(ir_path)
    if (not np.array_equal(rgb.video_ids, ir.video_ids) or not np.array_equal(rgb.video_ptr, ir.video_ptr)
            or not np.all(np.diff(rgb.video_ptr) == 20) or set(rgb.video_formats) != {"ffv1"} or set(ir.video_formats) != {"ffv1"}):
        raise ValueError("RGB/IR compact indexes are not position-aligned")
    return rgb, ir
