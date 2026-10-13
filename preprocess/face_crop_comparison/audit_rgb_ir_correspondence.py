"""Audit raw IR time coverage of RGB224 frames, without producing IR videos."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import RawRecording
from .process_full_dataset import source_signature


HERE = Path(__file__).resolve().parent
TOLERANCES_MS = (10., 20., 1000./30., 50., 100.)


def nearest_indices(query, reference):
    """Monotonic finite timestamps; exact ties choose the earlier frame."""
    if not len(reference) or not np.isfinite(query).all() or not np.isfinite(reference).all() or np.any(np.diff(reference) <= 0):
        raise ValueError("Invalid reference/query timestamps")
    right = np.searchsorted(reference, query)
    left = np.clip(right-1, 0, len(reference)-1)
    right = np.clip(right, 0, len(reference)-1)
    chosen = np.where(np.abs(reference[right]-query) < np.abs(reference[left]-query), right, left)
    return chosen, np.abs(reference[chosen]-query)*1000.


def load_rgb(directory):
    metadata = json.loads((directory / "face224_metadata.json").read_text())
    video = directory / "face224.mkv"
    if (metadata["status"] != "completed" or not metadata["verification"]["pixel_exact"]
            or video.stat().st_size != metadata["bytes"]):
        raise ValueError("RGB224 file or previous pixel verification is inconsistent")
    frames = pd.read_csv(directory / "face224_frames.csv", usecols=[
        "output_frame_index", "source_frame_index", "source_recorder_timestamp_unix",
    ], float_precision="round_trip")
    if (len(frames) != metadata["retained_frames"] or not len(frames)
            or not np.array_equal(frames.output_frame_index, np.arange(len(frames)))
            or np.any(np.diff(frames.source_frame_index) <= 0)
            or not np.isfinite(frames.source_recorder_timestamp_unix).all()
            or np.any(np.diff(frames.source_recorder_timestamp_unix) <= 0)):
        raise ValueError("RGB224 retained-frame mapping is inconsistent")
    return frames


def audit_video(path):
    directory = Path(path).parent
    video_id = f"{directory.parent.name.removesuffix('_data')}_{directory.name}"
    row = {"video_id": video_id, "mirror": directory.parent.name.removesuffix("_data"),
           "rgb_path": str(path), "rgb_retained_frames": 0, "ir_source_frames": 0,
           "ir_structure_status": "pending", "reason": ""}
    rgb = load_rgb(directory)
    row["rgb_retained_frames"] = len(rgb)
    ir_path = directory / "raw_ir_video.avi"
    if not ir_path.exists() or not ir_path.stat().st_size:
        return {**row, "ir_structure_status": "missing_or_empty_ir"}, np.empty(0)
    if not Path(str(ir_path)+".ts").exists():
        return {**row, "ir_structure_status": "missing_ir_timestamps"}, np.empty(0)
    recording = None
    try:
        signatures = {str(p): source_signature(p) for p in (ir_path, Path(str(ir_path)+".ts"))}
        recording = RawRecording(ir_path)
        ir_times = recording.timestamps
        rgb_times = rgb.source_recorder_timestamp_unix.to_numpy()
        chosen, distances = nearest_indices(rgb_times, ir_times)
        inverse, _ = nearest_indices(ir_times, rgb_times)
        mutual = inverse[chosen] == np.arange(len(rgb_times))
        row.update(ir_structure_status="consistent_jpeg_timestamp_index", ir_source_frames=len(ir_times),
                   first_ir_minus_first_retained_rgb_ms=float((ir_times[0]-rgb_times[0])*1000),
                   nearest_dt_ms_median=float(np.median(distances)), nearest_dt_ms_p95=float(np.quantile(distances, .95)),
                   nearest_dt_ms_max=float(distances.max()))
        for ms in TOLERANCES_MS:
            name = f"{ms:.3f}ms"
            valid = distances <= ms
            row[f"nearest_frames_{name}"] = int(valid.sum())
            row[f"unique_ir_frames_{name}"] = len(np.unique(chosen[valid]))
            row[f"mutual_nearest_frames_{name}"] = int((valid & mutual).sum())
        if any(source_signature(p) != signature for p, signature in signatures.items()):
            raise RuntimeError("Raw IR source changed during audit")
        return row, distances
    except (OSError, ValueError, pd.errors.ParserError) as error:
        row.update(ir_structure_status="unverified_ir_index", reason=str(error))
        return row, np.empty(0)
    finally:
        if recording is not None:
            recording.close()


def audit_reviewed_frames(preview, output):
    reviewed = pd.read_csv(preview / "tables/frame_audit.csv", float_precision="round_trip")
    rows = []
    for record in reviewed.to_dict("records"):
        directory = Path(record["source_path"]).parent
        row = {"video_id": record["video_id"], "ir_test_frame_index": record["source_frame_index"],
               "ir_accepted": record["direct_eligible"], "ir_reason": record["rejection_reason"],
               "rgb224_exists": (directory / "face224.mkv").exists(), "mutual_nearest": False,
               "delta_ms": np.nan, "ir_crop_exists": bool(record["direct_eligible"] and (preview / record["crop_path"]).is_file()),
               "preview_path": str(preview / record["preview_path"])}
        if row["rgb224_exists"]:
            rgb = load_rgb(directory)
            rgb_times = rgb.source_recorder_timestamp_unix.to_numpy()
            i, delta = nearest_indices(np.array([record["source_recorder_timestamp_unix"]]), rgb_times)
            i = int(i[0])
            ir_times = pd.read_csv(str(record["source_path"])+".ts", skipinitialspace=True, float_precision="round_trip").ts.to_numpy()
            j, _ = nearest_indices(np.array([rgb_times[i]]), ir_times)
            row.update(rgb_output_frame_index=int(rgb.iloc[i].output_frame_index),
                       rgb_source_frame_index=int(rgb.iloc[i].source_frame_index),
                       nearest_ir_source_frame_index=int(j[0]), delta_ms=float(delta[0]),
                       mutual_nearest=bool(j[0] == record["source_frame_index"]))
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(output / "tables/reviewed_frame_pairs.csv", index=False)
    counterexamples = paired.loc[paired.mutual_nearest & paired.delta_ms.le(1000/30) & ~paired.ir_accepted]
    counterexamples.to_csv(output / "tables/confirmed_quality_counterexamples.csv", index=False)
    return paired, counterexamples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/root/shared/HealthMirrorRawData"))
    parser.add_argument("--preview-dir", type=Path, default=HERE / "outputs/ir100_224")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/rgb_ir_correspondence")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new audit output directory")
    (args.output_dir / "tables").mkdir(parents=True)
    files = sorted(args.raw_root.glob("mirror*_data/patient_*/face224.mkv"))
    if not files:
        raise ValueError("No RGB224 videos")
    rows, distances = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(audit_video, str(path)) for path in files]
        for future in as_completed(futures):
            row, delta = future.result()
            rows.append(row)
            if len(delta):
                distances.append(delta)
            if len(rows) % 100 == 0:
                print(f"[rgb-ir-audit] {len(rows)}/{len(files)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("video_id")
    frame.to_csv(args.output_dir / "tables/video_correspondence.csv", index=False)
    summary = {"rgb224_videos": len(frame), "rgb224_retained_frames": int(frame.rgb_retained_frames.sum()),
               "ir_structure_status_videos": frame.ir_structure_status.value_counts().to_dict(),
               "ir_structure_status_rgb_frames": frame.groupby("ir_structure_status").rgb_retained_frames.sum().to_dict(),
               "matching_clock": "same-session absolute recorder Unix timestamps; never independent per-stream elapsed origins",
               "time_tolerances_ms": list(TOLERANCES_MS), "time_tolerance_is_final_protocol": False,
               "full_ir_quality_audit_performed": False, "rgb_videos_redecoded": False,
               "completed_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    all_distances = np.concatenate(distances) if distances else np.empty(0)
    summary["nearest_dt_ms_quantiles_verified_sources"] = (dict(zip(["min", "p50", "p95", "p99", "max"],
        np.quantile(all_distances, [0, .5, .95, .99, 1]).tolist())) if len(all_distances) else {})
    coverage = []
    for ms in TOLERANCES_MS:
        name = f"{ms:.3f}ms"
        n = int(frame[f"nearest_frames_{name}"].fillna(0).sum())
        unique = int(frame[f"unique_ir_frames_{name}"].fillna(0).sum())
        mutual = int(frame[f"mutual_nearest_frames_{name}"].fillna(0).sum())
        coverage.append({"tolerance_ms": ms, "rgb_nearest_raw_ir_frames": n,
                         "unique_raw_ir_frames": unique, "mutual_nearest_pairs": mutual,
                         "nearest_coverage_all_rgb": n/summary["rgb224_retained_frames"],
                         "mutual_coverage_all_rgb": mutual/summary["rgb224_retained_frames"]})
    pd.DataFrame(coverage).to_csv(args.output_dir / "tables/time_coverage.csv", index=False)
    paired, counterexamples = audit_reviewed_frames(args.preview_dir, args.output_dir)
    summary["reviewed_ir_frames_with_rgb_video"] = int(paired.rgb224_exists.sum())
    summary["reviewed_mutual_pairs_within_33ms"] = int((paired.mutual_nearest & paired.delta_ms.le(1000/30)).sum())
    summary["reviewed_mutual_pairs_ir_quality_rejected"] = len(counterexamples)
    summary["existing_processed_ir224_mkv_files"] = len(list(args.raw_root.glob("mirror*_data/patient_*/*ir*.mkv")))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# RGB224 / IR Correspondence Audit", "",
             f"RGB224: {len(frame):,} videos, {summary['rgb224_retained_frames']:,} retained frames.", "",
             "No full production IR quality masks exist yet. Raw-time coverage below is not IR224 quality eligibility.",
             "RGB mappings were checked against existing pixel-exact production metadata and file sizes; videos were not redecoded.",
             "Raw IR JPEG count must match its own monotonic timestamp sidecar. No ordinal-index or independent elapsed-origin fallback is allowed.", "",
             "|Raw IR source status|Videos|Affected RGB frames|", "|---|---:|---:|"]
    for status in summary["ir_structure_status_videos"]:
        lines.append(f"|{status}|{summary['ir_structure_status_videos'][status]}|{summary['ir_structure_status_rgb_frames'][status]:,}|")
    lines += ["", "## Time Coverage", "",
              "Tolerance sensitivity only, not a finalized cross-modal pairing protocol. Nearest coverage can reuse IR frames; mutual nearest gives unique, conservative pairs and is not maximum-cardinality matching.", "",
              "|Tolerance (ms)|RGB with nearest raw IR|Unique IR frames|Mutual nearest pairs|RGB nearest coverage|", "|---:|---:|---:|---:|---:|"]
    for row in coverage:
        lines.append(f"|{row['tolerance_ms']:.3f}|{row['rgb_nearest_raw_ir_frames']:,}|{row['unique_raw_ir_frames']:,}|{row['mutual_nearest_pairs']:,}|{row['nearest_coverage_all_rgb']:.2%}|")
    lines += ["", "## Existing Quality Evidence", "",
              f"Among the 100 reviewed IR frames, {summary['reviewed_mutual_pairs_within_33ms']} have a mutual-nearest retained RGB frame within 33.333 ms; {len(counterexamples)} of these IR frames fail the unchanged face-crop quality policy.",
              "This establishes that RGB eligibility does not imply nearest IR eligibility. It does not exclude the possibility of a different IR frame passing under a wider time tolerance, and it is not a full-dataset IR retention estimate.", "",
              "Full-video shared usability requires independently audited IR crops and a specified unique time-pair policy; use the RGB/IR quality-mask intersection, never copy RGB frame indices into IR.", "",
              "Individual counterexamples: tables/confirmed_quality_counterexamples.csv. Global per-video coverage: tables/video_correspondence.csv."]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(pd.DataFrame(coverage).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
