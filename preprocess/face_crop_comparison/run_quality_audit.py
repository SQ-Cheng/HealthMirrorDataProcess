"""Audit every frame of 100 raw videos, without creating video caches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import html
import json
import multiprocessing as mp
from pathlib import Path
import shutil

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.time_alignment import read_session_metadata
from .face_quality import FaceQualityGate, QualityConfig, clipped_crop_bounds
from .mediapipe_alignment import ComparisonConfig, KalmanComparison
from .pipeline import CropConfig, MediaPipeDetector, MODEL_DIR, RawRecording, VideoWriter
from .run_comparison import sha256


ROOT = Path(__file__).resolve().parent


def comparison_preview(frame, aligned, unfiltered, row, index):
    canvas = np.full((416, 896, 3), 248, np.uint8)
    original = frame.copy() if frame is not None else np.zeros((480, 640, 3), np.uint8)
    expanded = np.array([row.get(f"expanded_{key}", np.nan) for key in ("x1", "y1", "x2", "y2")])
    if np.isfinite(expanded).all():
        raw_box = expanded.copy()
        raw_box[1] = (expanded[1] + .2 * expanded[3]) / 1.2
        for box, color in ((raw_box, (40, 40, 230)), (expanded, (0, 165, 255))):
            coordinates = np.round(box).astype(int)
            cv2.rectangle(original, tuple(coordinates[:2]), tuple(coordinates[2:]), color, 2)
    filtered = np.array([row.get(f"crop_{key}", np.nan) for key in ("x1", "y1", "x2", "y2")])
    if np.isfinite(filtered).all():
        coordinates = np.round(filtered).astype(int)
        cv2.rectangle(original, tuple(coordinates[:2]), tuple(coordinates[2:]), (220, 200, 0), 2)
    height, width = original.shape[:2]
    scale = min(448 / width, 336 / height)
    resized = cv2.resize(original, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    left, top = (448 - resized.shape[1]) // 2, 32 + (336 - resized.shape[0]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    canvas[88:312, 448:672] = aligned
    canvas[88:312, 672:896] = unfiltered
    cv2.putText(canvas, "Raw: red detector / orange expanded / cyan Kalman", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (25, 25, 25), 1, cv2.LINE_AA)
    for x, label, valid in ((448, "Kalman + alignment", row["aligned_eligible"]),
                            (672, "No Kalman / alignment", row["unfiltered_direct_eligible"])):
        cv2.putText(canvas, label, (x + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, .40, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "valid" if valid else "rejected", (x + 4, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Frame {index} | {row['source_elapsed_seconds']:.3f}s | score={row.get('confidence', np.nan):.3f}",
                (5, 385), cv2.FONT_HERSHEY_SIMPLEX, .45, (25, 25, 25), 1, cv2.LINE_AA)
    reason = row["rejection_reason"]
    font_scale = min(.40, 875 / max(cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0], 1))
    cv2.putText(canvas, reason, (5, 405), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def select_videos(raw_root, output, seed):
    generator = np.random.default_rng(seed)
    selected, audit, used_patients = [], [], set()
    mirrors = sorted(raw_root.glob("mirror*_data"))
    if len(mirrors) != 5:
        raise ValueError(f"Expected five source mirrors, got {len(mirrors)}")
    for mirror in mirrors:
        paths = sorted(mirror.glob("patient_*/raw_video.avi"))
        count = 0
        for index in generator.permutation(len(paths)):
            path = paths[index]
            video_id = f"{mirror.name.removesuffix('_data')}_{path.parent.name}"
            entry = {"video_id": video_id, "raw_video_path": str(path)}
            try:
                metadata = read_session_metadata(path.parent / "patient_info.txt",
                                                 expected_local_id=path.parent.name.removeprefix("patient_"))
                hospital_id = metadata["session_hospital_id"]
                if hospital_id in used_patients:
                    audit.append({**entry, "selection_status": "duplicate_patient", "reason": ""})
                    continue
                recording = RawRecording(path)
                try:
                    middle = recording.frame(len(recording.timestamps) // 2)
                    if middle is None:
                        raise ValueError("Middle source frame failed to decode")
                    entry.update({"hospital_id": hospital_id,
                                  "mirror": mirror.name.removesuffix("_data"),
                                  "session_time_unix": metadata["session_time_unix"],
                                  "source_frames": len(recording.timestamps),
                                  "source_width": middle.shape[1], "source_height": middle.shape[0],
                                  "duration_seconds": float(recording.timestamps[-1] - recording.timestamps[0])})
                finally:
                    recording.close()
            except Exception as error:
                audit.append({**entry, "selection_status": "invalid_source", "reason": str(error)})
                continue
            selected.append(entry)
            used_patients.add(hospital_id)
            audit.append({**entry, "selection_status": "selected", "reason": ""})
            count += 1
            if count == 20:
                break
        if count != 20:
            raise RuntimeError(f"Only {count} eligible distinct patients in {mirror}")
    pd.DataFrame(selected).to_csv(output / "tables/selected_videos.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "tables/selection_audit.csv", index=False)
    return selected


def process_video(record, output_path, quality_config):
    cv2.setNumThreads(1)
    output = Path(output_path)
    source = Path(record["raw_video_path"])
    hashes = {str(path): sha256(path) for path in (source, Path(str(source) + ".ts"))}
    detector = MediaPipeDetector(CropConfig(confidence=.10))
    gate = FaceQualityGate(QualityConfig(**quality_config))
    processor = KalmanComparison(ComparisonConfig(), quality_gate=gate)
    recording = RawRecording(source)
    preview = VideoWriter(output / "videos" / f"{record['video_id']}.mp4", 896, 416, lossless=False)
    rows, snapshots, saved_reasons = [], [], set()
    print(f"[start] {record['video_id']} frames={len(recording.timestamps)}", flush=True)
    try:
        for index, timestamp in enumerate(recording.timestamps - recording.timestamps[0]):
            frame = recording.frame(index)
            if frame is None:
                row = {"status": "source_decode_failed", "quality_reason": "source_decode_failed",
                       "direct_eligible": False, "aligned_eligible": False, "candidate_count": 0}
                crops = {method: np.zeros((224, 224, 3), np.uint8) for method in ("direct", "aligned")}
            else:
                boxes, landmarks = detector.detect_with_keypoints(frame)
                crops, row = processor.process(frame, boxes, landmarks, float(timestamp))
                row["top_detection_confidence"] = float(boxes[:, 4].max()) if len(boxes) else np.nan
            if row["aligned_eligible"]:
                reason = "accepted"
            elif not row.get("quality_accepted", False):
                reason = row.get("quality_reason", row["status"])
            elif row.get("padding_fraction", 0) > .05:
                reason = "alignment_source_padding_over_5_percent"
            else:
                reason = row["alignment_status"]
            row.update({"video_id": record["video_id"], "hospital_id": record["hospital_id"],
                        "source_frame_index": index, "source_elapsed_seconds": float(timestamp),
                        "canonical_session_frame_time_unix": record["session_time_unix"] + float(timestamp),
                        "source_recorder_timestamp_unix": float(recording.timestamps[index]),
                        "rejection_reason": reason})
            unfiltered = np.zeros((224, 224, 3), np.uint8)
            row["unfiltered_direct_eligible"] = bool(row.get("raw_quality_accepted", False))
            if row["unfiltered_direct_eligible"]:
                x1, y1, x2, y2 = clipped_crop_bounds(frame, [row[f"expanded_{key}"] for key in ("x1", "y1", "x2", "y2")])
                row.update({f"unfiltered_crop_{key}": value for key, value in zip(("x1", "y1", "x2", "y2"), (x1, y1, x2, y2))})
                unfiltered = cv2.resize(frame[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_LINEAR)
            pair = comparison_preview(frame, crops["aligned"] if row["aligned_eligible"] else np.zeros_like(unfiltered),
                                      unfiltered, row, index)
            preview.write(pair, float(timestamp))
            rows.append(row)
            group = ("accepted" if reason == "accepted" else "low_confidence" if "low_confidence" in reason
                     else "outside_source" if "outside_source" in reason else "other_rejection")
            if frame is not None and (group not in saved_reasons or index == len(recording.timestamps) // 2):
                image_path = output / "figures/review" / f"{record['video_id']}_{index:05d}.jpg"
                cv2.imwrite(str(image_path), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
                snapshots.append({"path": str(image_path.relative_to(output)), "reason": reason,
                                  "source_frame_index": index})
                saved_reasons.add(group)
    finally:
        recording.close()
        preview.close()
        gate.close()
        detector.close()
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise RuntimeError(f"Source changed during audit: {source}")
    table = pd.DataFrame(rows)
    table.to_csv(output / "tables/frames" / f"{record['video_id']}.csv", index=False)
    Path(str(output / "videos" / f"{record['video_id']}.mp4") + ".ts").write_bytes(Path(str(source) + ".ts").read_bytes())
    confidence = table.get("confidence", pd.Series(np.nan, index=table.index))
    result = {**record, "source_hashes": hashes, "source_unchanged": True,
              "detected_frames": int(table.top_detection_confidence.notna().sum()),
              "selected_candidate_frames": int(confidence.notna().sum()),
              "direct_eligible_frames": int(table.direct_eligible.sum()),
              "aligned_eligible_frames": int(table.aligned_eligible.sum()),
              "unfiltered_direct_eligible_frames": int(table.unfiltered_direct_eligible.sum()),
              "mean_selected_confidence": float(confidence.mean()) if confidence.notna().any() else None,
              "snapshots": snapshots, "quality_config": quality_config}
    (output / "tables/frames" / f"{record['video_id']}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[done] {record['video_id']} accepted={result['aligned_eligible_frames']}/{len(table)}", flush=True)
    return result


def summarize(output, results, config, seed):
    tables = [pd.read_csv(output / "tables/frames" / f"{row['video_id']}.csv",
                          dtype={"hospital_id": str}) for row in results]
    frames = pd.concat(tables, ignore_index=True)
    verification = []
    for result, table in zip(results, tables):
        path = output / "videos" / f"{result['video_id']}.mp4"
        count, maximum_error = 0, 0.0
        with av.open(str(path)) as container:
            for decoded in container.decode(video=0):
                if count >= len(table) or decoded.width != 896 or decoded.height != 416:
                    raise RuntimeError(f"Preview frame/dimension mismatch: {path}")
                error = abs(float(decoded.time) - table.source_elapsed_seconds.iloc[count])
                maximum_error = max(maximum_error, error)
                pixels = decoded.to_ndarray(format="bgr24")
                for offset, mask in ((448, "aligned_eligible"), (672, "unfiltered_direct_eligible")):
                    mean = pixels[120:260, offset + 40:offset + 184].mean()
                    if (bool(table[mask].iloc[count]) and mean < 2) or (not bool(table[mask].iloc[count]) and mean > 2):
                        raise RuntimeError(f"Preview eligibility/content mismatch: {path}, frame {count}, {mask}")
                count += 1
        if count != len(table) or maximum_error > .00051:
            raise RuntimeError(f"Preview count/timestamp mismatch: {path}")
        if Path(str(path) + ".ts").read_bytes() != Path(result["raw_video_path"] + ".ts").read_bytes():
            raise RuntimeError(f"Timestamp sidecar differs: {path}")
        verification.append({"video_id": result["video_id"], "verified_frames": count,
                             "max_timestamp_error_seconds": maximum_error,
                             "branch_eligibility_content_check": True, "original_sidecar_exact": True})
    pd.DataFrame(verification).to_csv(output / "tables/preview_verification.csv", index=False)
    for column in ("confidence", "top_detection_confidence"):
        if column not in frames:
            frames[column] = np.nan
    summaries = [{key: value for key, value in row.items()
                  if key not in ("snapshots", "source_hashes", "quality_config")} for row in results]
    videos = pd.DataFrame(summaries)
    videos["aligned_retention_fraction"] = videos.aligned_eligible_frames / videos.source_frames
    videos.to_csv(output / "tables/video_summary.csv", index=False)
    counts = frames.rejection_reason.value_counts().rename_axis("reason").reset_index(name="frames")
    counts["fraction_all_frames"] = counts.frames / len(frames)
    counts.to_csv(output / "tables/rejection_counts.csv", index=False)
    flags = frames.rejection_reason.str.split(";").explode().value_counts().rename_axis("flag").reset_index(name="frames")
    flags.to_csv(output / "tables/rejection_flags_overlapping.csv", index=False)
    series = {"highest_detector_score": frames.top_detection_confidence.dropna(),
              "selected_candidate_score": frames.confidence.dropna(),
              "accepted_aligned_score": frames.loc[frames.aligned_eligible, "confidence"].dropna()}
    quantiles, histogram = [], []
    bins = np.linspace(0, 1, 21)
    for name, values in series.items():
        quantiles.append({"population": name, "frames": len(values), "mean": values.mean(),
                          "sd": values.std(), **{f"q{int(q * 100):02d}": values.quantile(q)
                                                for q in (0, .05, .25, .5, .75, .95, 1)}})
        frequencies, _ = np.histogram(values, bins)
        histogram.extend({"population": name, "lower": lower, "upper": upper,
                          "frames": int(count), "fraction_population": count / max(len(values), 1)}
                         for lower, upper, count in zip(bins[:-1], bins[1:], frequencies))
    pd.DataFrame(quantiles).to_csv(output / "tables/confidence_quantiles.csv", index=False)
    pd.DataFrame(histogram).to_csv(output / "tables/confidence_histogram.csv", index=False)
    sensitivity = [{"confidence_threshold": threshold,
                    "selected_frames_meeting_threshold": int(frames.confidence.ge(threshold).sum()),
                    "fraction_all_source_frames": float(frames.confidence.ge(threshold).mean())}
                   for threshold in (.1, .5, .6, .7, .8, .85, .9, .95)]
    pd.DataFrame(sensitivity).to_csv(output / "tables/confidence_threshold_sensitivity.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    colors = ("#747b85", "#197a9c", "#ce663b")
    labels = ("Highest detector score per frame", "Selected primary candidate", "Accepted for alignment")
    for (name, values), color, label in zip(series.items(), colors, labels):
        frequencies, _ = np.histogram(values, bins)
        axes[0].stairs(frequencies / max(len(values), 1), bins, color=color, label=label, linewidth=1.8)
        sorted_values = np.sort(values)
        if len(sorted_values):
            axes[1].plot(sorted_values, np.arange(1, len(values) + 1) / len(values), color=color, label=label)
    for axis in axes:
        axis.axvline(config["confidence_threshold"], ls="--", color="black", lw=1)
        axis.set(xlim=(0, 1), xlabel="BlazeFace detection score (not calibrated probability)")
        axis.grid(alpha=.2)
    axes[0].set(ylabel="Fraction within scored population", title="Frame-score distribution")
    axes[1].set(ylabel="Cumulative fraction", title="Empirical cumulative distribution")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "figures/confidence_distribution.png", dpi=180)
    fig.savefig(output / "figures/confidence_distribution.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plotted = counts.head(12).copy()
    if len(counts) > 12:
        plotted = pd.concat([plotted, pd.DataFrame([{"reason": "other", "frames": counts.iloc[12:].frames.sum()}])], ignore_index=True)
    plotted = plotted.sort_values("frames")
    short_names = {"accepted": "Accepted", "low_confidence": "Low score",
                   "expanded_box_outside_source_over_10_percent": "Expanded box over 10% outside image",
                   "ambiguous_faces": "Ambiguous primary face",
                   "association_rejected": "Temporal association rejected"}
    labels = plotted.reason.map(lambda value: " + ".join(short_names.get(flag, flag.replace("_", " ")) for flag in value.split(";")))
    axes[0].barh(labels, plotted.frames, color="#197a9c")
    axes[0].set(xlabel="Source frames", title="Mutually exclusive final outcomes")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[1].hist(videos.aligned_retention_fraction, bins=np.linspace(0, 1, 21), color="#ce663b", edgecolor="white")
    axes[1].set(xlabel="Accepted fraction of original frames", ylabel="Videos", title="Per-video retention", xlim=(0, 1))
    fig.savefig(output / "figures/quality_rejections.png", dpi=180)
    fig.savefig(output / "figures/quality_rejections.pdf")
    plt.close(fig)
    protocol = {"quality_gate": FaceQualityGate(QualityConfig(**config)).manifest(),
                "crop": KalmanComparison().manifest(), "detector_reporting_floor": .10,
                "models": {"mediapipe": json.loads((MODEL_DIR / "manifest.json").read_text())["mediapipe"]},
                "selection_seed": seed, "videos": len(results), "patients": videos.hospital_id.nunique(),
                "source_frames": len(frames), "aligned_eligible_frames": int(frames.aligned_eligible.sum()),
                "unfiltered_direct_eligible_frames": int(frames.unfiltered_direct_eligible.sum()),
                "preview_all_frames_count_timestamp_mask_checks_pass": True,
                "preview_layout": "Original with detection boxes | Kalman + alignment | no Kalman/alignment",
                "preview_dimensions": [896, 416],
                "selection": "20 distinct patients per mirror; no lab availability restriction; all source frames",
                "source_unchanged": all(row["source_unchanged"] for row in results),
                "confidence_missing": "No primary candidate => NaN, never filled with zero",
                "storage": "CSV, aggregate plots, review JPEGs and viewing-only paired MP4s; no training video cache"}
    (output / "manifest.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sections = []
    for result in results:
        sections.append(f'<section><h2>{html.escape(result["video_id"])}</h2>'
                        f'<p>Accepted {result["aligned_eligible_frames"]}/{result["source_frames"]} frames</p>'
                        f'<video controls preload="none" width="896" style="max-width:100%" src="videos/{result["video_id"]}.mp4"></video>'
                        + ''.join(f'<a href="{entry["path"]}"><img loading="lazy" width="768" '
                                  f'src="{entry["path"]}" alt="{html.escape(entry["reason"])}"></a>'
                                  for entry in result["snapshots"]) + '</section>')
    (output / "index.html").write_text('<!doctype html><html><meta charset="utf-8"><title>100-video face quality audit</title>'
                                      '<style>body{font:15px sans-serif;margin:24px}img{max-width:100%}'
                                      'section{border-top:1px solid #ddd;margin-top:24px}</style>'
                                      '<h1>MediaPipe pre-alignment quality audit</h1>'
                                      '<p>Videos: original with boxes | Kalman + alignment | no Kalman/alignment. '
                                      'Red: detector box; orange: top-expanded box; cyan: filtered crop. '
                                      'Rejected crops are black. Only confidence and expanded-box source containment '
                                      'are used for the pre-alignment quality gate.</p>'
                                      + ''.join(sections) + '</html>')
    report = ["# Pre-alignment quality audit", "",
              f"Videos: {len(results)}; distinct patients: {videos.hospital_id.nunique()}; source frames: {len(frames):,}.",
              f"Accepted for alignment: {int(frames.aligned_eligible.sum()):,} ({frames.aligned_eligible.mean():.2%}).", "",
              f"Accepted without Kalman/alignment: {int(frames.unfiltered_direct_eligible.sum()):,} "
              f"({frames.unfiltered_direct_eligible.mean():.2%}). "
              f"Videos with no accepted aligned frames: {int(videos.aligned_eligible_frames.eq(0).sum())}.", "",
              "Detection reporting floor: 0.10; selected-primary acceptance threshold: 0.80. "
              "Undetected/unselected frames have missing scores, not zero scores. Distributions are conditional "
              "on reported detections and frame-weighted, not patient-weighted. Threshold sensitivity is "
              "confidence-only, not a re-evaluation of source-boundary and temporal checks at each threshold.", "",
              "Expand the detected bbox upward by 20% of its original height. Reject it only if the expanded "
              "rectangle has more than 10% of its area outside [0,width] x [0,height]. The fraction is "
              "(expanded area - image intersection area) / expanded area; exactly 10% is allowed. "
              "Accepted boxes are clipped to the image for cropping, without padding. No extra "
              "2 px margin, face-oval coverage, mesh matching or core-landmark containment gate is applied. "
              "The confidence threshold remains 0.80, and primary-face ambiguity/association checks remain unchanged. "
              "The aligned branch still requires valid eye geometry and at most 5% source padding.", "",
              "Detector scores are uncalibrated, and bbox containment does not prove patient identity "
              "or that every facial feature is visible. Previous contour-based rejection has been removed.", "",
              "All source hashes verified unchanged. Paired MP4 videos are viewing previews, not training caches. "
              "Left: original video with red detector, orange expanded and cyan filtered boxes. "
              "Middle: quality-gated Kalman + alignment. Right: same detector, raw completeness/score gates and "
              "resize, but no Kalman and no alignment. Missing/rejected frames remain black at their source positions.", "",
              "| Outcome | Frames | Fraction |", "|---|---:|---:|"]
    report.extend(f"| {row.reason} | {row.frames:,} | {row.fraction_all_frames:.2%} |" for row in counts.itertuples())
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\n".join(report[:4]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("/root/shared/HealthMirrorRawData"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/quality100")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Rerun the same selected videos and replace outputs")
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.raw_root.resolve()):
        raise ValueError("Output must be outside source root")
    if args.resume and args.overwrite:
        raise ValueError("Choose resume or overwrite, not both")
    if args.output_dir.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(args.output_dir)
    selected = args.output_dir / "tables/selected_videos.csv"
    if args.overwrite:
        if not selected.exists():
            raise ValueError("Overwrite requires an existing video selection")
        for directory in ("tables/frames", "figures/review", "videos"):
            path = args.output_dir / directory
            if path.exists():
                shutil.rmtree(path)
        (args.output_dir / "figures/three_column_preview_example.png").unlink(missing_ok=True)
    for directory in ("tables/frames", "figures/review", "videos"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    records = (pd.read_csv(selected, dtype={"hospital_id": str}).to_dict("records")
               if (args.resume or args.overwrite) and selected.exists() else select_videos(args.raw_root, args.output_dir, args.seed))
    config = asdict(QualityConfig())
    results, pending = [], []
    for record in records:
        metadata = args.output_dir / "tables/frames" / f"{record['video_id']}.json"
        if args.resume and metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["quality_config"] != config or any(sha256(path) != value for path, value in saved["source_hashes"].items()):
                raise RuntimeError("Resume source/configuration mismatch")
            results.append(saved)
        else:
            pending.append(record)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(process_video, record, str(args.output_dir), config) for record in pending]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"[progress] {len(results)}/{len(records)}", flush=True)
    summarize(args.output_dir, sorted(results, key=lambda row: row["video_id"]), config, args.seed)


if __name__ == "__main__":
    main()
