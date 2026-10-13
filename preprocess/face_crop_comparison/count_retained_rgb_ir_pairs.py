"""Filter the frozen mutual-nearest population by full IR quality masks."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import av
import numpy as np
import pandas as pd

from .audit_rgb_ir_correspondence import nearest_indices
from .pipeline import RawRecording
from .process_full_dataset import source_signature
from .run_comparison import sha256


HERE = Path(__file__).resolve().parent
MAX_DELTA_MS = 1000./30.
FRAMES = 20
GAP = 2
QUANTILES = np.linspace(.05, .95, FRAMES)


def select20(rgb_indices, ir_indices):
    """Use existing quantiles, uniform fallback, then a feasible nonadjacent chain."""
    n = len(rgb_indices)
    if n != len(ir_indices) or np.any(np.diff(rgb_indices) <= 0) or np.any(np.diff(ir_indices) <= 0):
        raise ValueError("Pair source indices must increase in both modalities")
    chain = []
    for position in range(n):
        if not chain or (rgb_indices[position]-rgb_indices[chain[-1]] >= GAP
                         and ir_indices[position]-ir_indices[chain[-1]] >= GAP):
            chain.append(position)
    if len(chain) < FRAMES:
        return np.empty(0, dtype=int), len(chain)

    def valid(selected):
        return (len(np.unique(selected)) == FRAMES and np.min(np.diff(rgb_indices[selected])) >= GAP
                and np.min(np.diff(ir_indices[selected])) >= GAP)

    for selected in (np.rint(QUANTILES*(n-1)).astype(int), np.rint(np.linspace(0, n-1, FRAMES)).astype(int)):
        if valid(selected):
            return selected, len(chain)
    chain = np.asarray(chain)
    selected = chain[np.rint(np.linspace(0, len(chain)-1, FRAMES)).astype(int)]
    if not valid(selected):
        raise AssertionError("Feasible nonadjacent chain failed 20-frame selection")
    return selected, len(chain)


def retained_mapping(directory, prefix):
    metadata = json.loads((directory / f"{prefix}_metadata.json").read_text())
    frames = pd.read_csv(directory / f"{prefix}_frames.csv", float_precision="round_trip")
    n = metadata["retained_frames"]
    if metadata["status"] not in ("completed", "no_valid_frames") or len(frames) != n:
        raise ValueError("Incomplete processed frame metadata")
    if (not np.array_equal(frames.output_frame_index, np.arange(n))
            or np.any(np.diff(frames.source_frame_index) <= 0)):
        raise ValueError("Invalid output/source frame ordering")
    if n:
        if (not metadata["verification"]["pixel_exact"] or (directory / f"{prefix}.mkv").stat().st_size != metadata["bytes"]
                or not np.isfinite(frames.source_recorder_timestamp_unix).all()
                or np.any(np.diff(frames.source_recorder_timestamp_unix) <= 0)
                or frames.confidence.isna().any() or frames.confidence.lt(.75).any()):
            raise ValueError("Processed video or accepted-frame quality is inconsistent")
    return frames, metadata


def verify_selected(path, mapping):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != "ffv1" or (stream.width, stream.height) != (224, 224):
            raise ValueError("Expected native224 FFV1 video")
        for row in mapping.itertuples(index=False):
            target = int(row.encoded_pts)
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for packet in container.demux(stream):
                if not packet.size or packet.pts is None or packet.pts < target:
                    continue
                if packet.pts != target or not packet.is_keyframe or str(packet.time_base) != row.encoded_time_base:
                    raise ValueError("Selected packet timestamp/keyframe mismatch")
                decoded = packet.decode()
                if len(decoded) != 1 or decoded[0].to_ndarray(format="bgr24").shape != (224, 224, 3):
                    raise ValueError("Selected frame failed native224 decoding")
                break
            else:
                raise ValueError("Selected FFV1 frame is missing")


def count_video(job):
    baseline, ir_record = job["baseline"], job["ir_record"]
    video_id = baseline["video_id"]
    directory = Path(baseline["rgb_path"]).parent
    ir_directory = Path(job["ir_root"]) / directory.relative_to(job["raw_root"])
    rgb, rgb_metadata = retained_mapping(directory, "face224")
    if len(rgb) != baseline["rgb_retained_frames"]:
        raise RuntimeError(f"RGB baseline changed: {video_id}")
    expected = int(baseline.get("mutual_nearest_frames_33.333ms", 0) or 0)
    row = {"video_id": video_id, "mirror": baseline["mirror"], "rgb_frames": len(rgb),
           "candidate_pairs": expected, "retained_pairs": 0, "ir_status": ir_record["status"],
           "max_nonadjacent_pairs": 0, "pairs20_eligible": False,
           "rgb_video_path": str(directory / "face224.mkv"), "ir_video_path": "",
           "quality_unknown_candidate_pairs": 0, "reason": ""}
    if baseline["ir_structure_status"] != "consistent_jpeg_timestamp_index":
        return row, [], {}
    recording = RawRecording(directory / "raw_ir_video.avi")
    try:
        ir_times = recording.timestamps
        rgb_times = rgb.source_recorder_timestamp_unix.to_numpy()
        nearest, distances = nearest_indices(rgb_times, ir_times)
        inverse, _ = nearest_indices(ir_times, rgb_times)
        candidate = np.flatnonzero((inverse[nearest] == np.arange(len(rgb))) & (distances <= MAX_DELTA_MS))
        if len(candidate) != expected:
            raise RuntimeError(f"Frozen mutual-nearest population changed: {video_id}: {len(candidate)} != {expected}")
        if ir_record["status"] not in ("completed", "no_valid_frames"):
            row.update(quality_unknown_candidate_pairs=expected, reason=ir_record.get("error", "IR output incomplete"))
            return row, [], {}
        ir, metadata = retained_mapping(ir_directory, "face224_ir")
        if (metadata["source_frames"] != len(ir_times)
                or any(source_signature(path) != signature for path, signature in metadata["source_signatures"].items())):
            raise RuntimeError(f"IR source/output changed: {video_id}")
        positions = ir.source_frame_index.to_numpy(np.int64)
        if len(positions) and (positions.min() < 0 or positions.max() >= len(ir_times)):
            raise ValueError("IR accepted source index out of range")
        if not np.allclose(ir.source_recorder_timestamp_unix.to_numpy(np.float64), ir_times[positions], rtol=0, atol=1e-6):
            raise ValueError("IR accepted timestamps disagree with raw source")
        lookup = np.full(len(ir_times), -1, dtype=np.int64)
        lookup[positions] = ir.output_frame_index
        keep = lookup[nearest[candidate]] >= 0
        rgb_rows = candidate[keep]
        ir_rows = lookup[nearest[rgb_rows]]
        pairs = pd.DataFrame({"rgb_output_frame_index": rgb_rows, "ir_output_frame_index": ir_rows,
                              "rgb_source_frame_index": rgb.source_frame_index.to_numpy()[rgb_rows],
                              "ir_source_frame_index": nearest[rgb_rows],
                              "rgb_recorder_timestamp_unix": rgb_times[rgb_rows],
                              "ir_recorder_timestamp_unix": ir_times[nearest[rgb_rows]],
                              "pair_delta_ms": distances[rgb_rows],
                              "rgb_canonical_session_frame_time_unix": rgb.canonical_session_frame_time_unix.to_numpy()[rgb_rows]})
        if pairs.ir_source_frame_index.duplicated().any() or pairs.rgb_source_frame_index.duplicated().any():
            raise AssertionError("A paired source frame was reused")
        selected, maximum = select20(pairs.rgb_source_frame_index.to_numpy(), pairs.ir_source_frame_index.to_numpy())
        row.update(retained_pairs=len(pairs), max_nonadjacent_pairs=maximum, pairs20_eligible=len(selected) == FRAMES,
                   ir_video_path=str(ir_directory / "face224_ir.mkv") if len(ir) else "")
        selected_pairs = pairs.iloc[selected].copy()
        if len(selected):
            verify_selected(directory / "face224.mkv", rgb.iloc[rgb_rows[selected]])
            verify_selected(ir_directory / "face224_ir.mkv", ir.iloc[ir_rows[selected]])
            selected_pairs.insert(0, "video_id", video_id)
            selected_pairs.insert(1, "sample_pair_index", np.arange(FRAMES))
        audit = pd.read_csv(ir_directory / "face224_ir_audit.csv", usecols=["source_frame_index", "rejection_reason"])
        if len(audit) != len(ir_times) or not np.array_equal(audit.source_frame_index, np.arange(len(ir_times))):
            raise ValueError("IR frame audit is incomplete")
        reasons = Counter(audit.rejection_reason.to_numpy()[nearest[candidate[~keep]]])
        return row, selected_pairs.to_dict("records"), dict(reasons)
    finally:
        recording.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/root/shared/HealthMirrorRawData"))
    parser.add_argument("--ir-root", type=Path)
    parser.add_argument("--baseline-dir", type=Path, default=HERE / "outputs/rgb_ir_correspondence")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/rgb_ir_paired224")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.ir_root = args.ir_root or args.raw_root
    baseline = pd.read_csv(args.baseline_dir / "tables/video_correspondence.csv").fillna({"mutual_nearest_frames_33.333ms": 0})
    expected_total = int(baseline['mutual_nearest_frames_33.333ms'].sum())
    if expected_total != 1980915:
        raise RuntimeError(f"Expected the requested frozen 1,980,915 candidates, found {expected_total}")
    current_rgb = {str(path) for path in args.raw_root.glob("mirror*_data/patient_*/face224.mkv")}
    if current_rgb != set(baseline.rgb_path):
        raise RuntimeError("The RGB video cohort changed since the frozen correspondence audit")
    if args.preflight:
        print(f"[paired-preflight] RGB videos={len(baseline)} frozen candidates={expected_total}", flush=True)
        return
    index = pd.read_csv(args.ir_root / "_face224_ir_processing/index.csv", keep_default_na=False)
    if index.video_id.duplicated().any() or index.status.eq("pending").any():
        raise RuntimeError("Full IR processing has not reached a terminal status for every session")
    records = index.set_index("video_id").to_dict("index")
    if not set(baseline.video_id).issubset(records):
        raise RuntimeError("IR run did not account for the complete RGB cohort")
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "RUNNING.json").write_text(json.dumps({"candidate_pairs": expected_total, "started_utc": pd.Timestamp.now(tz="UTC").isoformat()})+"\n")
    rows, selected, reasons = [], [], Counter()
    jobs = [{"baseline": row, "ir_record": records[row["video_id"]], "ir_root": str(args.ir_root), "raw_root": str(args.raw_root)}
            for row in baseline.to_dict("records")]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(count_video, job) for job in jobs]
        for future in as_completed(futures):
            row, samples, rejected = future.result()
            rows.append(row); selected.extend(samples); reasons.update(rejected)
            if len(rows) % 100 == 0:
                print(f"[paired224] {len(rows)}/{len(jobs)} retained_pairs={sum(r['retained_pairs'] for r in rows)}", flush=True)
    table = pd.DataFrame(rows).sort_values("video_id")
    if table.candidate_pairs.sum() != expected_total:
        raise AssertionError("Frozen candidate totals changed")
    table.to_csv(args.output_dir / "tables/video_pair_counts.csv", index=False)
    selected_table = pd.DataFrame(selected)
    if len(selected_table):
        selected_table = selected_table.sort_values(["video_id", "sample_pair_index"])
    else:
        selected_table = pd.DataFrame(columns=["video_id", "sample_pair_index", "rgb_source_frame_index", "ir_source_frame_index"])
    selected_table.to_csv(args.output_dir / "tables/selected20_pairs.csv", index=False)
    mirror = table.groupby("mirror").agg(rgb_videos=("video_id", "size"), candidate_pairs=("candidate_pairs", "sum"),
        retained_pairs=("retained_pairs", "sum"), videos_with_pairs=("retained_pairs", lambda x: int(x.gt(0).sum())),
        videos_at_least20_pairs=("retained_pairs", lambda x: int(x.ge(FRAMES).sum())),
        eligible20_videos=("pairs20_eligible", "sum")).reset_index()
    mirror.to_csv(args.output_dir / "tables/mirror_pair_counts.csv", index=False)
    pd.DataFrame([{"reason": reason, "candidate_pairs_removed": count} for reason, count in reasons.items()]).to_csv(args.output_dir / "tables/ir_rejection_reasons.csv", index=False)
    kept = int(table.retained_pairs.sum())
    unknown = int(table.quality_unknown_candidate_pairs.sum())
    if sum(reasons.values()) != expected_total-kept-unknown:
        raise AssertionError("IR rejection reasons do not account for the frozen candidate population")
    summary = {"candidate_pairs": expected_total, "retained_rgb_ir_pairs": kept,
               "retained_fraction": kept/expected_total, "not_retained_pairs": expected_total-kept,
               "confirmed_ir_quality_rejected_pairs": expected_total-kept-unknown,
               "quality_unknown_candidate_pairs": unknown,
               "complete_for_candidate_population": unknown == 0,
               "rgb_cohort_videos": len(table), "videos_with_retained_pairs": int(table.retained_pairs.gt(0).sum()),
               "videos_with_at_least20_pairs": int(table.retained_pairs.ge(FRAMES).sum()),
               "eligible_20frame_rgb_ir_videos": int(table.pairs20_eligible.sum()),
               "selected20_pair_count": len(selected), "max_pair_time_delta_ms": MAX_DELTA_MS,
               "minimum_raw_frame_gap_each_modality": GAP, "ir_processing_statuses": index.status.value_counts().to_dict(),
               "ir_quality_rejection_reasons": dict(reasons),
               "pair_policy": "Freeze original RGB-retained/raw-IR mutual-nearest candidates first; quality filter only; never rematch",
               "selection_policy": "5-95% quantiles; uniform fallback; feasible nonadjacent chain fallback to avoid losing valid videos",
               "selected_frames_verified_by_decoding_both_videos": True,
               "baseline_sha256": sha256(args.baseline_dir / "tables/video_correspondence.csv"),
               "completed_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(mirror))
    for offset, column, label, color in ((-.18, "candidate_pairs", "Frozen candidates", "#87939a"),
                                         (.18, "retained_pairs", "Quality-retained pairs", "#287b69")):
        axes[0].bar(x+offset, mirror[column], width=.35, label=label, color=color)
    for offset, column, label, color in ((-.18, "rgb_videos", "RGB224 videos", "#397ca5"),
                                         (.18, "eligible20_videos", "Paired 20-frame videos", "#287b69")):
        axes[1].bar(x+offset, mirror[column], width=.35, label=label, color=color)
    for ax in axes:
        ax.set_xticks(x, mirror.mirror)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set(title="Fixed mutual-nearest population", ylabel="RGB+IR frame pairs")
    axes[1].set(title="Unique, nonadjacent 20-pair eligibility", ylabel="Videos")
    figure.tight_layout()
    figure.savefig(figures / "pair_retention_summary.png", dpi=180)
    plt.close(figure)
    lines = ["# Quality-Filtered RGB224 + IR224 Pairs", "", f"Frozen candidates: {expected_total:,}.",
             f"Retained pairs: {kept:,} ({kept/expected_total:.2%}).",
             f"Videos with any retained pair: {summary['videos_with_retained_pairs']:,}.",
             f"Videos with at least 20 pairs: {summary['videos_with_at_least20_pairs']:,}.",
             f"Eligible 20-frame paired videos: {summary['eligible_20frame_rgb_ir_videos']:,}.",
             f"Candidate pairs with unknown IR quality because processing failed: {unknown:,}.", "",
             "Mutual nearest was calculated against all raw IR timestamps before filtering IR quality. Accepted outputs must be fresh .75-confidence Kalman crops, <=10% expanded-box overflow, without alignment. No neighbor replacement, repeated IR frame or held-frame fallback is allowed.", "",
             "20-frame eligibility requires both raw source-index gaps >=2, unique frames, and successful decoding of all selected RGB/IR 224 FFV1 frames. Quantiles and a feasible-chain fallback spread samples without unnecessarily discarding sparse valid sequences.", "",
             "Counts cover the full physical RGB video cohort, not a lab-matched, valid-patient, admission, surgery or task-specific cohort. Clinical pairing time is the existing validated RGB canonical time; IR-specific clinical origins are not invented.", "",
             "|Mirror|RGB videos|Candidate pairs|Retained pairs|20-frame paired videos|", "|---|---:|---:|---:|---:|"]
    for row in mirror.itertuples(index=False):
        lines.append(f"|{row.mirror}|{row.rgb_videos}|{row.candidate_pairs:,}|{row.retained_pairs:,}|{row.eligible20_videos}|")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines)+"\n")
    (args.output_dir / "RUNNING.json").unlink()
    print(json.dumps(summary, indent=2), flush=True)
    if unknown:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
