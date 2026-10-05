"""Compare direct vs aligned MediaPipe crops on the same 20 original videos."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import html
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import hashlib

from .mediapipe_alignment import ComparisonConfig, KalmanComparison
from .pipeline import CropConfig, MediaPipeDetector, RawRecording, VideoWriter, MODEL_DIR
from .run_comparison import sha256, verify_lossless


EXP_DIR = Path(__file__).resolve().parent
METHODS = ("direct", "aligned")


def make_preview(raw, crops, row, elapsed):
    canvas = np.full((368, 896, 3), 244, np.uint8)
    original = raw.copy()
    if "crop_x1" in row:
        cv2.rectangle(original, (row["crop_x1"], row["crop_y1"]),
                      (row["crop_x2"], row["crop_y2"]), (185, 155, 20), 2)
    if "alignment_m00" in row:
        matrix = np.array([[row[f"alignment_m{r}{c}"] for c in range(3)] for r in range(2)])
        corners = np.array([[[0, 0], [223, 0], [223, 223], [0, 223]]], dtype=np.float32)
        polygon = cv2.transform(corners, cv2.invertAffineTransform(matrix)).astype(np.int32)
        cv2.polylines(original, polygon, True, (40, 115, 230), 2)
    canvas[32:, :448] = cv2.resize(original, (448, 336), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, f"Raw RGB | {elapsed:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (35, 35, 35), 1, cv2.LINE_AA)
    for index, method in enumerate(METHODS):
        left = 448 + index * 224
        canvas[64:288, left:left + 224] = crops[method]
        cv2.putText(canvas, f"{method} + Kalman", (left + 5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (35, 35, 35), 1, cv2.LINE_AA)
        status = row["status"] if method == "direct" else row["alignment_status"]
        cv2.putText(canvas, status, (left + 5, 315), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (35, 35, 35), 1, cv2.LINE_AA)
        if method == "aligned" and "alignment_angle_degrees" in row:
            cv2.putText(canvas, f"roll: {row['alignment_angle_degrees']:.1f} deg", (left + 5, 338),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def process_video(record, output_path, config_dict):
    cv2.setNumThreads(1)
    output = Path(output_path)
    directory = output / "videos" / record["video_id"]
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(record["raw_video_path"])
    original_hashes = {str(path): sha256(path) for path in (source, Path(str(source) + ".ts"))}
    recording = RawRecording(source)
    processor = KalmanComparison(ComparisonConfig(**config_dict))
    detector = MediaPipeDetector(CropConfig())
    elapsed = recording.timestamps - recording.timestamps[0]
    hashes = {method: hashlib.sha256() for method in METHODS}
    selected = set(np.round(np.array([0.1, 0.5, 0.9]) * (len(elapsed) - 1)).astype(int))
    rows, stills, writers = [], [], {}
    print(f"[video-start] {record['video_id']} frames={len(elapsed)}", flush=True)
    try:
        for method in METHODS:
            writers[method] = VideoWriter(directory / f"{method}_224.mkv", 224, 224)
        writers["preview"] = VideoWriter(directory / "comparison.mp4", 896, 368, lossless=False)
        for index, timestamp in enumerate(elapsed):
            frame = recording.frame(index)
            if frame is None:
                crops = {method: np.zeros((224, 224, 3), np.uint8) for method in METHODS}
                row = {"status": "source_decode_failed", "alignment_status": "source_decode_failed",
                       "direct_eligible": False, "aligned_eligible": False}
            else:
                boxes, landmarks = detector.detect_with_keypoints(frame)
                crops, row = processor.process(frame, boxes, landmarks, float(timestamp))
            row.update({"source_frame_index": index,
                        "source_elapsed_seconds": timestamp,
                        "source_recorder_timestamp_unix": recording.timestamps[index],
                        "canonical_session_frame_time_unix": record["session_time_unix"] + timestamp,
                        "paired_eligible": row["direct_eligible"] and row["aligned_eligible"]})
            rows.append(row)
            for method in METHODS:
                hashes[method].update(crops[method].tobytes())
                writers[method].write(crops[method], float(timestamp))
            preview = make_preview(frame if frame is not None else np.zeros((480, 640, 3), np.uint8),
                                   crops, row, timestamp)
            writers["preview"].write(preview, float(timestamp))
            if index in selected:
                path = output / "figures" / f"{record['video_id']}_frame_{index:05d}.png"
                cv2.imwrite(str(path), preview)
                stills.append(str(path.relative_to(output)))
    finally:
        for writer in writers.values():
            writer.close()
        detector.close()
        recording.close()
    records = pd.DataFrame(rows)
    records.to_csv(directory / "frame_records.csv", index=False)
    summaries = []
    for method in METHODS:
        path = directory / f"{method}_224.mkv"
        checked = verify_lossless(path, hashes[method].hexdigest(), elapsed)
        Path(str(path) + ".ts").write_bytes(Path(str(source) + ".ts").read_bytes())
        summaries.append({"video_id": record["video_id"], "method": method,
                          "source_frames": len(records),
                          "eligible_frames": int(records[f"{method}_eligible"].sum()),
                          "paired_eligible_frames": int(records["paired_eligible"].sum()),
                          "output_bytes": path.stat().st_size, **checked})
    if any(sha256(path) != expected for path, expected in original_hashes.items()):
        raise RuntimeError(f"Original source changed: {source}")
    result = {**record, "source_hashes": original_hashes, "source_unchanged": True,
              "protocol": processor.manifest(), "summaries": summaries, "stills": stills}
    (directory / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[video-complete] {record['video_id']} paired_frames={records.paired_eligible.sum()}", flush=True)
    return result


def build_review(output, results, selection_dir):
    summaries = pd.DataFrame([row for result in results for row in result["summaries"]])
    summaries.to_csv(output / "tables" / "method_summary.csv", index=False)
    totals = summaries.groupby("method")[["source_frames", "eligible_frames", "paired_eligible_frames", "output_bytes"]].sum()
    totals.to_csv(output / "tables" / "aggregate_summary.csv")
    notes_path = selection_dir / "tables" / "manual_review_notes.csv"
    notes = pd.read_csv(notes_path).set_index("video_id")["review_note"].to_dict() if notes_path.exists() else {}
    sections = []
    sheet = Image.new("RGB", (1792, 1330), "white")
    draw = ImageDraw.Draw(sheet)
    quality = []
    for index, result in enumerate(results):
        video_id = result["video_id"]
        rows = pd.read_csv(output / "videos" / video_id / "frame_records.csv")
        padding = rows.get("padding_fraction", pd.Series(np.nan, index=rows.index))
        quality.append({"video_id": video_id, "source_frames": len(rows),
                        "direct_eligible_fraction": float(rows.direct_eligible.mean()),
                        "paired_eligible_fraction": float(rows.paired_eligible.mean()),
                        "alignment_padding_over_5_percent_frames": int(padding.gt(.05).sum()),
                        "manual_review_note": notes.get(video_id, "")})
        middle = cv2.imread(str(output / result["stills"][1]))
        left, top = (index % 4) * 448, (index // 4) * 266
        sheet.paste(Image.fromarray(cv2.cvtColor(middle[64:288, 448:896], cv2.COLOR_BGR2RGB)), (left, top + 40))
        draw.text((left + 8, top + 3), video_id, fill="black")
        draw.text((left + 8, top + 22), "Direct + Kalman", fill="black")
        draw.text((left + 232, top + 22), "Alignment + Kalman", fill="black")
        sections.append(f'<section><h2>{html.escape(video_id)}</h2>'
                        f'<p>{html.escape(notes.get(video_id, ""))}</p>'
                        f'<video controls preload="none" width="896" src="videos/{video_id}/comparison.mp4"></video>'
                        + ''.join(f'<a href="{path}"><img loading="lazy" width="448" src="{path}"></a>' for path in result["stills"])
                        + '</section>')
    sheet.save(output / "figures" / "contact_sheet_20.png")
    pd.DataFrame(quality).to_csv(output / "tables" / "quality_flags.csv", index=False)
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>MediaPipe Kalman alignment comparison</title>'
        '<style>body{font:16px sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;}'
        'section{border-top:1px solid #ccc;margin-top:30px;}video,img{max-width:100%;height:auto;}'
        'img{margin:8px 12px 8px 0;}</style><h1>MediaPipe: direct vs aligned, both Kalman filtered</h1>'
        '<p>One detector pass and identical Kalman-filtered rectangular bbox per frame. '
        'Top margin: 20% original bbox height. Direct resize vs eye-level roll alignment. '
        'Both output 224 x 224 RGB; FFV1 MKV is lossless, MP4 is a viewing preview. '
        'Inspect the patient face and paired eligibility before training.</p>' + ''.join(sections) + '</html>'
    )
    manifest = {"videos": len(results), "patients": len({row["hospital_id"] for row in results}),
                "protocol": KalmanComparison().manifest(),
                "detector": json.loads((MODEL_DIR / "manifest.json").read_text())["mediapipe"],
                "package_versions": {name: version(name) for name in ("mediapipe", "av", "numpy", "opencv-python")},
                "all_lossless_checks_pass": True, "source_unchanged": True,
                "selection_reference": str(selection_dir), "insightface_used": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# MediaPipe direct/alignment comparison", "",
             "Same 20 patients and complete recordings as the prior pilot. Shared MediaPipe detections, "
             "20% top-only bbox expansion, and four normalized-coordinate Kalman filters. "
             "The aligned branch adds a Kalman-filtered eye-line rotation around the shared bbox center. "
             "Both branches map the rectangular bbox extent to 224 x 224 with bilinear interpolation. "
             "No side/bottom expansion or square ROI is used.", "",
             "| Method | Source frames | Candidate eligible frames | Paired eligible frames |",
             "|---|---:|---:|---:|"]
    for method, row in totals.iterrows():
        lines.append(f"| {method} | {int(row.source_frames):,} | {int(row.eligible_frames):,} | {int(row.paired_eligible_frames):,} |")
    lines.extend(["", "All 40 FFV1 files passed full-frame pixel/count/timestamp verification; original hashes were unchanged.",
                  "Alignment corrects in-plane head tilt. It does not reconstruct side-view or out-of-frame anatomy. "
                  "Aligned frames with more than 5% source-border padding are ineligible. Missing/held frames retain "
                  "source positions and are ineligible. For a controlled comparison use paired_eligible.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    print(totals.to_string(), flush=True)
    print(f"[review-ready] {output / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, default=EXP_DIR / "outputs/pilot20")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "outputs/mediapipe_kalman_alignment20")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output_dir}")
    if args.output_dir.resolve().is_relative_to(Path("/root/shared/HealthMirrorRawData")):
        raise ValueError("Output must be outside the source data directory")
    for subdirectory in ("videos", "figures", "tables"):
        (args.output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    selection = args.selection_dir / "tables/selected_videos.csv"
    records = pd.read_csv(selection, dtype={"hospital_id": str}).to_dict("records")
    if len(records) != 20:
        raise ValueError("Expected exactly 20 reference videos")
    (args.output_dir / "tables/selected_videos.csv").write_bytes(selection.read_bytes())
    config = asdict(ComparisonConfig())
    results, pending = [], []
    for record in records:
        path = args.output_dir / "videos" / record["video_id"] / "metadata.json"
        if args.resume and path.exists():
            saved = json.loads(path.read_text())
            if saved["protocol"]["parameters"] != config or any(sha256(p) != value for p, value in saved["source_hashes"].items()):
                raise RuntimeError(f"Resume source/config mismatch: {record['video_id']}")
            results.append(saved)
        else:
            pending.append(record)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(process_video, row, str(args.output_dir), config) for row in pending]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"[progress] {len(results)}/20 verified", flush=True)
    results.sort(key=lambda row: row["video_id"])
    build_review(args.output_dir, results, args.selection_dir)


if __name__ == "__main__":
    main()
