"""Process 20 original recordings with both detectors and build a local review."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import html
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import time

import av
import cv2
import numpy as np
import pandas as pd

from study.common.time_alignment import read_session_metadata
from .pipeline import (CropConfig, FaceCropper, InsightFaceDetector,
                       MODEL_DIR, MediaPipeDetector, RawRecording, VideoWriter,
                       intersection_over_union)


EXP_DIR = Path(__file__).resolve().parent
REFERENCE = Path("study/exp2_face_pretrained_head32_regression/outputs/20frame/task_records")
BACKENDS = ("mediapipe", "insightface")
COLORS = {"mediapipe": (185, 155, 20), "insightface": (40, 115, 230)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while payload := handle.read(1024 * 1024):
            digest.update(payload)
    return digest.hexdigest()


def select_videos(raw_root, output, seed):
    frames = [pd.read_csv(path, dtype={"hospital_id": str},
                         usecols=["hospital_id", "video_id", "mirror", "lab_patient_id"])
              for path in sorted(REFERENCE.glob("*.csv"))]
    if not frames:
        raise RuntimeError(f"Missing validated patient/video reference: {REFERENCE}")
    records = pd.concat(frames, ignore_index=True)
    if records.groupby("video_id")["hospital_id"].nunique().gt(1).any():
        raise ValueError("A reference video belongs to multiple patients")
    records = records.drop_duplicates("video_id").sort_values("video_id")
    generator = np.random.default_rng(seed)
    chosen, audit, used_patients = [], [], set()
    for mirror_dir in sorted(raw_root.glob("mirror*_data")):
        mirror = mirror_dir.name.removesuffix("_data")
        candidates = records.loc[records["mirror"].eq(mirror)]
        selected_count = 0
        for index in generator.permutation(len(candidates)):
            candidate = candidates.iloc[index].to_dict()
            patient_id = candidate["hospital_id"].lstrip("0") or "0"
            path = mirror_dir / f"patient_{int(candidate['lab_patient_id']):06d}" / "raw_video.avi"
            entry = {"video_id": candidate["video_id"], "raw_video_path": str(path)}
            if patient_id in used_patients:
                continue
            try:
                metadata = read_session_metadata(
                    path.parent / "patient_info.txt",
                    expected_local_id=candidate["lab_patient_id"],
                    expected_hospital_id=candidate["hospital_id"],
                )
                recording = RawRecording(path)
                try:
                    middle = recording.frame(len(recording.ranges) // 2)
                    if middle is None:
                        raise ValueError("The middle source frame is corrupt")
                    candidate.update({
                        "raw_video_path": str(path), "source_frames": len(recording.ranges),
                        "duration_seconds": float(recording.timestamps[-1] - recording.timestamps[0]),
                        "source_width": middle.shape[1], "source_height": middle.shape[0],
                        "session_time_unix": metadata["session_time_unix"],
                        "session_time_local": metadata["session_time_local"].isoformat(),
                    })
                finally:
                    recording.close()
            except Exception as exc:
                audit.append({**entry, "status": "excluded", "reason": str(exc)})
                continue
            audit.append({**entry, "status": "selected", "reason": ""})
            chosen.append(candidate)
            used_patients.add(patient_id)
            selected_count += 1
            if selected_count == 4:
                break
        if selected_count != 4:
            raise RuntimeError(f"Only {selected_count} eligible distinct patients in {mirror}")
    if len(chosen) != 20:
        raise RuntimeError(f"Expected 20 videos from 5 mirrors; found {len(chosen)}")
    pd.DataFrame(chosen).to_csv(output / "tables" / "selected_videos.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "tables" / "selection_audit.csv", index=False)
    return chosen


def preview_frame(raw, crops, rows, elapsed):
    canvas = np.full((368, 896, 3), 244, np.uint8)
    original = raw.copy()
    for backend in BACKENDS:
        row = rows[backend]
        if "crop_x" in row:
            x, y, side = row["crop_x"], row["crop_y"], row["crop_side"]
            cv2.rectangle(original, (x, y), (x + side, y + side), COLORS[backend], 2)
    canvas[32:368, :448] = cv2.resize(original, (448, 336), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"Raw RGB | {elapsed:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (35, 35, 35), 1, cv2.LINE_AA)
    for number, backend in enumerate(BACKENDS):
        left = 448 + number * 224
        canvas[64:288, left:left + 224] = crops[backend]
        cv2.putText(canvas, backend, (left + 8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLORS[backend], 1, cv2.LINE_AA)
        cv2.putText(canvas, rows[backend]["status"], (left + 5, 315),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def verify_lossless(path, expected_hash, timestamps):
    digest = hashlib.sha256()
    count = 0
    max_time_error = 0.0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            digest.update(frame.to_ndarray(format="bgr24").tobytes())
            if count >= len(timestamps):
                raise RuntimeError(f"Extra output frame in {path}")
            max_time_error = max(max_time_error, abs(float(frame.time) - timestamps[count]))
            count += 1
    if count != len(timestamps) or digest.hexdigest() != expected_hash:
        raise RuntimeError(f"Lossless pixel/frame-count verification failed: {path}")
    if max_time_error > 0.00051:
        raise RuntimeError(f"Video PTS differs from source elapsed time: {path}")
    return {"lossless_pixel_check": "exact_all_frames", "verified_frames": count,
            "max_container_timestamp_error_seconds": max_time_error}


def process_video(record, output_dir, config_dict, gpu):
    cv2.setNumThreads(1)
    output = Path(output_dir)
    directory = output / "videos" / record["video_id"]
    directory.mkdir(parents=True, exist_ok=True)
    config = CropConfig(**config_dict)
    source = Path(record["raw_video_path"])
    source_hash = sha256(source)
    ts_hash = sha256(str(source) + ".ts")
    recording = RawRecording(source)
    detectors, writers = {}, {}
    frame_records, stills = [], []
    hashes = {backend: hashlib.sha256() for backend in BACKENDS}
    detector_seconds = {backend: 0.0 for backend in BACKENDS}
    cropper = {backend: FaceCropper(config) for backend in BACKENDS}
    elapsed = recording.timestamps - recording.timestamps[0]
    selected_frames = set(np.round(np.array([0.10, 0.50, 0.90]) * (len(elapsed) - 1)).astype(int))
    started = time.perf_counter()
    print(f"[video-start] {record['video_id']} frames={len(elapsed)} gpu={gpu}", flush=True)
    try:
        detectors["mediapipe"] = MediaPipeDetector(config)
        detectors["insightface"] = InsightFaceDetector(config, gpu=gpu)
        for backend in BACKENDS:
            writers[backend] = VideoWriter(directory / f"{backend}_224.mkv", 224, 224)
        writers["preview"] = VideoWriter(directory / "comparison.mp4", 896, 368, lossless=False)
        for index, timestamp in enumerate(elapsed):
            raw = recording.frame(index)
            row = {"source_frame_index": index, "source_recorder_timestamp_unix": recording.timestamps[index],
                   "source_elapsed_seconds": timestamp,
                   "canonical_session_frame_time_unix": record["session_time_unix"] + timestamp,
                   "source_byte_start": recording.ranges[index][0],
                   "source_byte_end": recording.ranges[index][1]}
            crops, crop_rows = {}, {}
            for backend in BACKENDS:
                if raw is None:
                    crop = np.zeros((224, 224, 3), np.uint8)
                    detail = {"status": "source_decode_failed", "train_eligible": False,
                              "candidate_count": 0}
                else:
                    before = time.perf_counter()
                    boxes = detectors[backend].detect(raw)
                    detector_seconds[backend] += time.perf_counter() - before
                    crop, detail = cropper[backend].crop(raw, boxes, float(timestamp))
                row.update({f"{backend}_{key}": value for key, value in detail.items()})
                crops[backend], crop_rows[backend] = crop, detail
                hashes[backend].update(crop.tobytes())
                writers[backend].write(crop, float(timestamp))
            if all("crop_x" in crop_rows[name] for name in BACKENDS):
                boxes = []
                for name in BACKENDS:
                    detail = crop_rows[name]
                    boxes.append(np.array([detail["crop_x"], detail["crop_y"],
                                           detail["crop_x"] + detail["crop_side"],
                                           detail["crop_y"] + detail["crop_side"]]))
                row["crop_iou"] = intersection_over_union(*boxes)
            frame_records.append(row)
            preview = preview_frame(
                raw if raw is not None else np.zeros((480, 640, 3), np.uint8),
                crops, crop_rows, timestamp,
            )
            writers["preview"].write(preview, float(timestamp))
            if index in selected_frames:
                image_path = output / "figures" / f"{record['video_id']}_frame_{index:05d}.png"
                cv2.imwrite(str(image_path), preview)
                stills.append(str(image_path.relative_to(output)))
            if (index + 1) % 500 == 0:
                print(f"[frames] {record['video_id']} {index + 1}/{len(elapsed)}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        for detector in detectors.values():
            detector.close()
        recording.close()
    records = pd.DataFrame(frame_records)
    for backend in BACKENDS:
        for field in ("crop_side", "upsampled"):
            column = f"{backend}_{field}"
            if column not in records:
                records[column] = False if field == "upsampled" else np.nan
    if "crop_iou" not in records:
        records["crop_iou"] = np.nan
    records.to_csv(directory / "frame_records.csv", index=False)
    summaries = []
    for backend in BACKENDS:
        video = directory / f"{backend}_224.mkv"
        (directory / f"{backend}_224.mkv.ts").write_bytes(Path(str(source) + ".ts").read_bytes())
        checked = verify_lossless(video, hashes[backend].hexdigest(), elapsed)
        eligible = records[f"{backend}_train_eligible"].astype(bool)
        summaries.append({
            "video_id": record["video_id"], "backend": backend,
            "source_frames": len(records), "train_eligible_frames": int(eligible.sum()),
            "train_eligible_fraction": float(eligible.mean()),
            "held_frames": int(records[f"{backend}_status"].eq("held_short_gap").sum()),
            "multi_face_frames": int(records[f"{backend}_candidate_count"].gt(1).sum()),
            "median_crop_side_pixels": float(records.loc[eligible, f"{backend}_crop_side"].median()),
            "upsampled_valid_frames": int(records.loc[eligible, f"{backend}_upsampled"].eq(True).sum()),
            "detector_seconds": detector_seconds[backend],
            "output_bytes": video.stat().st_size, **checked,
        })
    source_unchanged = sha256(source) == source_hash and sha256(str(source) + ".ts") == ts_hash
    if not source_unchanged:
        raise RuntimeError(f"Original source changed during processing: {source}")
    metadata = {**record, "config": config_dict, "source_sha256": source_hash,
                "source_timestamp_sha256": ts_hash, "source_unchanged": True,
                "insightface_model_class": detectors["insightface"].model_class,
                "insightface_providers": detectors["insightface"].providers,
                "output_codec": "FFV1 level 3 / bgr0 / no chroma subsampling",
                "source_clock_policy": "Session Timestamp + relative raw recorder elapsed time",
                "raw_timestamps_are_preserved": True,
                "preview_policy": "lossy H.264 for viewing only; never a training source",
                "total_seconds": time.perf_counter() - started,
                "stills": stills, "summaries": summaries}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[video-complete] {record['video_id']} seconds={metadata['total_seconds']:.1f}", flush=True)
    return metadata


def build_review(output, results, config):
    summaries = pd.DataFrame([summary for result in results for summary in result["summaries"]])
    summaries.to_csv(output / "tables" / "detector_summary.csv", index=False)
    notes_path = output / "tables" / "manual_review_notes.csv"
    notes = (pd.read_csv(notes_path).set_index("video_id")["review_note"].to_dict()
             if notes_path.is_file() else {})
    comparisons, quality_flags, blocks = [], [], []
    for result in results:
        video_id = result["video_id"]
        rows = pd.read_csv(output / "videos" / video_id / "frame_records.csv")
        common = rows["mediapipe_train_eligible"] & rows["insightface_train_eligible"]
        quality = {"video_id": video_id}
        reasons = []
        for backend in BACKENDS:
            selected = rows[f"{backend}_train_eligible"]
            coverage = float(selected.mean())
            x1, y1, x2, y2 = (rows.get(f"{backend}_box_{name}", pd.Series(np.nan, index=rows.index))
                               for name in ("x1", "y1", "x2", "y2"))
            outside = (x1.lt(0) | y1.lt(0) | x2.gt(result["source_width"])
                       | y2.gt(result["source_height"]))
            fraction = float(outside.loc[selected].mean()) if selected.any() else np.nan
            quality[f"{backend}_detected_fraction"] = coverage
            quality[f"{backend}_box_outside_source_fraction"] = fraction
            if coverage < 0.90:
                reasons.append(f"{backend}_detected_coverage_below_90_percent")
            if fraction > 0.10:
                reasons.append(f"{backend}_box_exceeds_source_in_over_10_percent_of_detected_frames")
        quality["manual_review_required"] = bool(reasons)
        quality["reasons"] = ";".join(reasons)
        quality["manual_review_note"] = notes.get(video_id, "")
        if quality["manual_review_note"]:
            quality["manual_review_required"] = True
        quality_flags.append(quality)
        comparisons.append({"video_id": video_id, "source_frames": len(rows),
                            "both_eligible_frames": int(common.sum()),
                            "median_crop_iou": rows.loc[common, "crop_iou"].median()})
        qc_text = ("Quality review required: " + html.escape(quality["reasons"])) if reasons else ""
        qc_text += " " + html.escape(quality["manual_review_note"])
        blocks.append(f'<section><h2>{html.escape(video_id)}</h2><p>{qc_text}</p>'
                      f'<video controls preload="none" width="896" '
                      f'src="videos/{video_id}/comparison.mp4"></video>'
                      f'<p>Full RGB / MediaPipe / InsightFace. '
                      f'Common detected frames: {common.sum()} / {len(rows)}.</p>'
                      + ''.join(f'<a href="{path}"><img loading="lazy" src="{path}" width="448"></a>'
                                for path in result["stills"])
                      + '</section>')
    pd.DataFrame(comparisons).to_csv(output / "tables" / "paired_comparison.csv", index=False)
    pd.DataFrame(quality_flags).to_csv(output / "tables" / "video_quality_flags.csv", index=False)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Original RGB face-crop comparison</title><style>'
            'body{font:16px sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;}'
            'section{border-top:1px solid #ccc;margin-top:30px;padding-top:10px;}'
            'video,img{max-width:100%;height:auto;}img{margin:8px 12px 8px 0;}'
            '</style><h1>Original RGB face-crop comparison: 20 recordings</h1>'
            '<p>MediaPipe BlazeFace short-range vs InsightFace SCRFD, identical crop rules. '
            'Training files: *_224.mkv (lossless FFV1). MP4 files are viewing previews. '
            'Missing detections remain at their original frame position; short held crops '
            'are excluded from training eligibility. Eligibility marks a detector candidate, '
            'not verified patient identity. Review flagged videos before training.</p>'
            + ''.join(blocks) + '</html>')
    (output / "index.html").write_text(page)
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (4 * 448, 5 * 266), "white")
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        image = cv2.imread(str(output / result["stills"][1]))
        left, top = (index % 4) * 448, (index // 4) * 266
        pair = image[64:288, 448:896]
        sheet.paste(Image.fromarray(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)), (left, top + 40))
        draw.text((left + 8, top + 3), result["video_id"], fill="black")
        draw.text((left + 8, top + 22), "MediaPipe", fill="#148fb0")
        draw.text((left + 232, top + 22), "InsightFace", fill="#d65e29")
    sheet.save(output / "figures" / "contact_sheet_20.png")
    aggregate = summaries.groupby("backend").agg(
        source_frames=("source_frames", "sum"), train_eligible_frames=("train_eligible_frames", "sum"),
        upsampled_valid_frames=("upsampled_valid_frames", "sum"),
        detector_seconds=("detector_seconds", "sum"), output_bytes=("output_bytes", "sum"),
    )
    aggregate["eligible_fraction"] = aggregate.train_eligible_frames / aggregate.source_frames
    aggregate["upsampled_fraction_of_eligible"] = aggregate.upsampled_valid_frames / aggregate.train_eligible_frames
    aggregate.to_csv(output / "tables" / "aggregate_summary.csv")
    manifest = {"config": asdict(config), "videos": len(results),
                "patients": len({result["hospital_id"] for result in results}),
                "selection": "4 per mirror, distinct patients, seeded eligible-reference session sampling",
                "all_lossless_checks_pass": all(summary["lossless_pixel_check"] == "exact_all_frames"
                                                for result in results for summary in result["summaries"]),
                "source_unchanged": all(result["source_unchanged"] for result in results),
                "models": json.loads((MODEL_DIR / "manifest.json").read_text()),
                "package_versions": {name: version(name) for name in (
                    "mediapipe", "insightface", "onnxruntime-gpu", "av", "numpy", "opencv-python"
                )}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Original-video face-crop comparison", "",
             f"Twenty complete recordings from 20 distinct patients; "
             f"{int(aggregate.source_frames.iloc[0]):,} original frames per detector.", "",
             "| Detector | Eligible frames | Eligible rate | Upsampled eligible frames | Saved FFV1 GiB |",
             "|---|---:|---:|---:|---:|"]
    for backend, row in aggregate.iterrows():
        lines.append(f"| {backend} | {int(row.train_eligible_frames):,} | "
                     f"{row.eligible_fraction:.2%} | {int(row.upsampled_valid_frames):,} "
                     f"({row.upsampled_fraction_of_eligible:.2%}) | "
                     f"{row.output_bytes / 1024**3:.2f} |")
    lines.extend(["", "All 40 FFV1 videos passed full-frame, pixel-exact BGR decoding checks. "
                  "Source video and timestamp hashes were unchanged. Output frame IDs match "
                  "the original timestamp CSVs; no frames were dropped or resampled.", "",
                  "Use `index.html` for the 20 side-by-side videos, and "
                  "`figures/contact_sheet_20.png` for a middle-frame overview. "
                  "Full-resolution detector crops are in `videos/`, not the MP4 previews.", "",
                  "The pilot uses identical crop rules for both detectors. Inspect forehead/chin "
                  "coverage, patient selection and stability across time before choosing a detector. "
                  "Coverage and ROI overlap are not manually labeled detector accuracy. "
                  "Short held crops are excluded from training eligibility.", "",
                  "Use `tables/video_quality_flags.csv` to find low-coverage and source-boundary "
                  "cases. The per-frame eligibility flag identifies a detector candidate; it "
                  "does not establish patient identity or replace video-level quality review.", ""])
    for video_id, note in notes.items():
        if video_id in {result["video_id"] for result in results}:
            lines.extend([f"Visual review of `{video_id}`: {note}", ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    print(aggregate.to_string(), flush=True)
    print(f"[review-ready] {output / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("/root/shared/HealthMirrorRawData"))
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "outputs" / "pilot20")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.output_dir.resolve().is_relative_to(args.raw_root.resolve()):
        raise ValueError("Output must be outside the original data directory")
    for subdirectory in ("tables", "figures", "videos"):
        (args.output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "tables" / "selected_videos.csv"
    selected = (
        pd.read_csv(selection_path, dtype={"hospital_id": str}).to_dict("records")
        if args.resume and selection_path.is_file()
        else select_videos(args.raw_root, args.output_dir, args.seed)
    )
    config = CropConfig()
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    results = []
    pending = []
    for record in selected:
        path = args.output_dir / "videos" / record["video_id"] / "metadata.json"
        if args.resume and path.is_file():
            saved = json.loads(path.read_text())
            if saved["config"] != asdict(config) or saved["source_sha256"] != sha256(record["raw_video_path"]):
                raise RuntimeError(f"Resume source/config changed: {record['video_id']}")
            results.append(saved)
        else:
            pending.append(record)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        jobs = {executor.submit(process_video, row, str(args.output_dir), asdict(config),
                                gpu_ids[index % len(gpu_ids)]): row
                for index, row in enumerate(pending)}
        for future in as_completed(jobs):
            results.append(future.result())
            print(f"[progress] {len(results)}/20 recordings verified", flush=True)
    results.sort(key=lambda row: row["video_id"])
    build_review(args.output_dir, results, config)


if __name__ == "__main__":
    main()
