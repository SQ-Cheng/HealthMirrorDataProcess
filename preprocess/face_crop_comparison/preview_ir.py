"""Review raw IR crops using the unchanged native224 RGB production geometry."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import html
from importlib.metadata import version
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from study.common.time_alignment import read_session_metadata
from .face_quality import FaceQualityGate, QualityConfig
from .mediapipe_alignment import ComparisonConfig, KalmanComparison
from .pipeline import CropConfig, MediaPipeDetector, MODEL_DIR, RawRecording
from .process_full_dataset import source_signature
from .run_comparison import sha256


HERE = Path(__file__).resolve().parent
DETECTOR = None
QUALITY = QualityConfig(confidence_threshold=.75)
GEOMETRY = ComparisonConfig(enable_alignment=False)


def target_frame(timestamps, seconds):
    return min(int(np.searchsorted(timestamps - timestamps[0], seconds)), len(timestamps)-1)


def select_sources(raw_root, count, seed, seconds):
    paths = sorted(raw_root.glob("mirror*_data/patient_*/raw_ir_video.avi"))
    mirrors = sorted({path.parent.parent.name for path in paths})
    if not mirrors:
        raise ValueError("No raw_ir_video.avi files found")
    rng = np.random.default_rng(seed)
    selected, audit = [], []
    for mirror_index, mirror in enumerate(mirrors):
        quota = count//len(mirrors) + (mirror_index < count % len(mirrors))
        pool = [path for path in paths if path.parent.parent.name == mirror]
        accepted = 0
        for index in rng.permutation(len(pool)):
            if accepted == quota:
                break
            path = pool[index]
            video_id = f"{mirror.removesuffix('_data')}_{path.parent.name}"
            row = {"video_id": video_id, "mirror": mirror.removesuffix("_data"), "source_path": str(path)}
            recording = None
            try:
                signatures = {str(p): source_signature(p) for p in (path, Path(str(path)+".ts"))}
                if path.stat().st_size == 0:
                    raise ValueError("Empty raw IR video")
                recording = RawRecording(path)
                frame_index = target_frame(recording.timestamps, seconds)
                frame = recording.frame(frame_index)
                if frame is None:
                    raise ValueError("Selected raw IR frame failed to decode")
                row.update({"source_frame_index": frame_index, "source_frames": len(recording.timestamps),
                            "source_width": frame.shape[1], "source_height": frame.shape[0],
                            "source_elapsed_seconds": float(recording.timestamps[frame_index]-recording.timestamps[0]),
                            "source_recorder_timestamp_unix": float(recording.timestamps[frame_index]),
                            "source_signatures": signatures, "hospital_id": "", "metadata_error": ""})
                try:
                    metadata = read_session_metadata(path.parent / "patient_info.txt",
                                                     expected_local_id=path.parent.name.removeprefix("patient_"))
                    row["hospital_id"] = metadata["session_hospital_id"]
                except Exception as error:
                    # Face-only inspection is independent of clinical metadata validity.
                    row["metadata_error"] = str(error)
                audit.append({**{k: v for k, v in row.items() if k != "source_signatures"},
                              "selection_status": "selected", "selection_reason": "readable_source; no face-score filtering"})
                selected.append(row)
                accepted += 1
            except (OSError, ValueError, pd.errors.ParserError) as error:
                audit.append({**row, "selection_status": "invalid_source", "selection_reason": str(error)})
            finally:
                if recording is not None:
                    recording.close()
        if accepted != quota:
            raise ValueError(f"Only {accepted}/{quota} readable sources in {mirror}")
    for index, row in enumerate(selected, 1):
        row["sample_id"] = index
    return selected, pd.DataFrame(audit), Counter(path.parent.parent.name for path in paths)


def make_preview(frame, crop, row, sample_id, video_id):
    canvas = np.full((416, 704, 3), 248, np.uint8)
    original = frame.copy()
    expanded = np.array([row.get(f"expanded_{key}", np.nan) for key in ("x1", "y1", "x2", "y2")])
    if np.isfinite(expanded).all():
        raw = expanded.copy()
        raw[1] = (expanded[1]+QUALITY.forehead_margin*expanded[3])/(1+QUALITY.forehead_margin)
        for box, color in ((raw, (40, 40, 230)), (expanded, (0, 165, 255))):
            box = np.round(box).astype(int)
            cv2.rectangle(original, tuple(box[:2]), tuple(box[2:]), color, 2)
    filtered = np.array([row.get(f"crop_{key}", np.nan) for key in ("x1", "y1", "x2", "y2")])
    if np.isfinite(filtered).all():
        box = np.round(filtered).astype(int)
        cv2.rectangle(original, tuple(box[:2]), tuple(box[2:]), (220, 200, 0), 2)
    scale = min(448/frame.shape[1], 336/frame.shape[0])
    resized = cv2.resize(original, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    left, top = (448-resized.shape[1])//2, 32+(336-resized.shape[0])//2
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
    accepted = bool(row["direct_eligible"])
    # Never display a held/rejected crop as if it were an eligible training frame.
    canvas[88:312, 464:688] = crop if accepted else np.zeros((224, 224, 3), np.uint8)
    cv2.putText(canvas, "Original IR: red detector / orange +20% / cyan Kalman", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .38, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Kalman / no alignment", (465, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "224 x 224 | accepted" if accepted else "REJECTED", (470, 335),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (35, 120, 35) if accepted else (40, 40, 210), 1, cv2.LINE_AA)
    caption = f"{sample_id:03d} {video_id} | frame {row['source_frame_index']} | {row['source_elapsed_seconds']:.3f}s | score={row.get('confidence', np.nan):.3f}"
    cv2.putText(canvas, caption, (5, 384), cv2.FONT_HERSHEY_SIMPLEX, .38, (25, 25, 25), 1, cv2.LINE_AA)
    reason = row["rejection_reason"]
    font = min(.40, 690/max(cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0], 1))
    cv2.putText(canvas, reason, (5, 406), cv2.FONT_HERSHEY_SIMPLEX, font, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def initialize_worker():
    global DETECTOR
    cv2.setNumThreads(1)
    DETECTOR = MediaPipeDetector(CropConfig(confidence=.10))
    # Multiprocessing finalizers run before Python shuts down MediaPipe threads.
    Finalize(None, DETECTOR.close, exitpriority=10)


def process_one(record, output):
    output = Path(output)
    source = Path(record["source_path"])
    before = {path: sha256(path) for path in record["source_signatures"]}
    if any(source_signature(path) != signature for path, signature in record["source_signatures"].items()):
        raise RuntimeError(f"Source changed after selection: {source}")
    recording = RawRecording(source)
    gate = FaceQualityGate(QUALITY)
    processor = KalmanComparison(GEOMETRY, gate)
    warmup_accepted = 0
    try:
        for index in range(record["source_frame_index"]+1):
            frame = recording.frame(index)
            elapsed = float(recording.timestamps[index]-recording.timestamps[0])
            if frame is None:
                crops = {"direct": np.zeros((224, 224, 3), np.uint8)}
                row = {"direct_eligible": False, "status": "source_decode_failed", "rejection_reason": "source_decode_failed"}
            else:
                boxes, landmarks = DETECTOR.detect_with_keypoints(frame)
                crops, row = processor.process(frame, boxes, landmarks, elapsed)
                row["top_detection_confidence"] = float(boxes[:, 4].max()) if len(boxes) else np.nan
                row["rejection_reason"] = ("accepted" if row["direct_eligible"] else
                    row.get("quality_reason", row["status"]) if not row.get("quality_accepted", False) else row["status"])
            warmup_accepted += int(row["direct_eligible"])
        if frame is None or elapsed != record["source_elapsed_seconds"]:
            raise RuntimeError("Selected frame/timestamp changed or failed to decode")
        stem = f"{record['sample_id']:03d}_{record['video_id']}_frame_{index:05d}"
        row.update({k: v for k, v in record.items() if k != "source_signatures"})
        row["processed_prefix_frames"] = index+1
        row["accepted_prefix_frames"] = warmup_accepted
        row["source_channels_identical"] = bool(np.array_equal(frame[..., 0], frame[..., 1]) and np.array_equal(frame[..., 1], frame[..., 2]))
        row["preview_path"] = f"figures/frames/{stem}.png"
        row["crop_path"] = ""
        if row["direct_eligible"]:
            crop = crops["direct"]
            row["crop_path"] = f"crops/{stem}.png"
            if not cv2.imwrite(str(output / row["crop_path"]), crop):
                raise RuntimeError("Could not save IR crop")
            if not np.array_equal(cv2.imread(str(output / row["crop_path"])), crop):
                raise RuntimeError("IR crop PNG failed pixel-exact validation")
            if row["source_channels_identical"] and not (np.array_equal(crop[..., 0], crop[..., 1]) and np.array_equal(crop[..., 1], crop[..., 2])):
                raise RuntimeError("Preprocessing unexpectedly changed grayscale channels")
        preview = make_preview(frame, crops["direct"], row, record["sample_id"], record["video_id"])
        if not cv2.imwrite(str(output / row["preview_path"]), preview):
            raise RuntimeError("Could not save IR preview")
        after = {path: sha256(path) for path in before}
        if before != after or any(source_signature(path) != signature for path, signature in record["source_signatures"].items()):
            raise RuntimeError("Raw IR video or timestamps changed during the test")
        row["source_sha256"] = before[str(source)]
        row["timestamps_sha256"] = before[str(source)+".ts"]
        row["original_files_unchanged"] = True
        return row
    finally:
        gate.close()
        recording.close()


def write_gallery(output, rows):
    cards = []
    for row in rows:
        label = f"{row['sample_id']:03d} | {row['video_id']} | {row['rejection_reason']}"
        path = html.escape(row["preview_path"], quote=True)
        cards.append(f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{path}"><img loading="lazy" src="{path}" width="704" height="416"></a></figure>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Raw IR face crop review</title><style>body{font-family:Arial,sans-serif;margin:24px;background:#fafafa;color:#222}'
            '.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,704px),1fr));gap:20px}'
            'figure{margin:0}img{max-width:100%;height:auto}figcaption{margin-bottom:6px}</style>'
            '<h1>Raw IR: 224 face crop review</h1><div class="samples">'+"\n".join(cards)+'</div></html>')
    (output / "index.html").write_text(page)
    for start in range(0, len(rows), 20):
        selected = rows[start:start+20]
        sheet = np.full((208*int(np.ceil(len(selected)/4)), 352*4, 3), 248, np.uint8)
        for index, row in enumerate(selected):
            image = cv2.imread(str(output / row["preview_path"]))
            y, x = (index//4)*208, (index % 4)*352
            sheet[y:y+208, x:x+352] = cv2.resize(image, (352, 208), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(output / f"figures/contact_sheet_{start//20+1:02d}.png"), sheet):
            raise RuntimeError("Could not save contact sheet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/root/shared/HealthMirrorRawData"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/ir100_224")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--elapsed-seconds", type=float, default=1.)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or not np.isfinite(args.elapsed_seconds) or args.elapsed_seconds < 0:
        raise ValueError("Invalid sample count, worker count or frame time")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; existing previews are not overwritten")
    for subdir in ("figures/frames", "crops", "tables"):
        (args.output_dir / subdir).mkdir(parents=True)
    selected, audit, inventory = select_sources(args.raw_root, args.count, args.seed, args.elapsed_seconds)
    pd.DataFrame([{k: v for k, v in row.items() if k != "source_signatures"} for row in selected]).to_csv(args.output_dir / "tables/selected_sources.csv", index=False)
    audit.to_csv(args.output_dir / "tables/source_selection_audit.csv", index=False)
    manifest = {"raw_root": str(args.raw_root), "raw_ir_inventory": dict(inventory), "count": args.count,
                "selection_seed": args.seed, "target_elapsed_seconds": args.elapsed_seconds,
                "sampling": "equal counts per mirror, distinct sessions, no face-score selection",
                "quality": asdict(QUALITY), "geometry": asdict(GEOMETRY), "detector_reporting_floor": .10,
                "kalman_state": "process every preceding source frame from frame zero, never skip context frames",
                "ir_handling": "decode source MJPEG; replicate grayscale for MediaPipe; no intensity normalization, RGB bbox transfer or alignment",
                "timestamp_policy": "IR-specific .ts must exactly match JPEG count and be monotonic; no positional fallback",
                "clinical_timestamp_assignment": False, "outputs": "100 preview PNGs including rejections; only accepted crops saved as pixel-exact PNG",
                "mediapipe_version": version("mediapipe"), "opencv_version": cv2.__version__,
                "detector_weight_sha256": sha256(MODEL_DIR / "blaze_face_short_range.tflite"),
                "implementation_sha256": {name: sha256(HERE / name) for name in ("pipeline.py", "face_quality.py", "mediapipe_alignment.py", "preview_ir.py")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker) as pool:
        futures = [pool.submit(process_one, record, str(args.output_dir)) for record in selected]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0:
                print(f"[ir-preview] {len(rows)}/{args.count}; accepted={sum(row['direct_eligible'] for row in rows)}", flush=True)
    rows.sort(key=lambda row: row["sample_id"])
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "tables/frame_audit.csv", index=False)
    write_gallery(args.output_dir, rows)
    summary = {"preview_frames": len(rows), "accepted_frames": int(frame.direct_eligible.sum()),
               "rejected_frames": int((~frame.direct_eligible).sum()), "reasons": frame.rejection_reason.value_counts().to_dict(),
               "source_validation_attempts": len(audit), "skipped_invalid_sources": int(audit.selection_status.eq("invalid_source").sum()),
               "processed_prefix_frames": int(frame.processed_prefix_frames.sum()), "original_files_unchanged": bool(frame.original_files_unchanged.all()),
               "confidence_quantiles": frame.confidence.dropna().quantile([0, .1, .5, .9, 1]).to_dict(),
               "completed_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output_dir / "REPORT.md").write_text("# Raw IR 224 Review\n\n"+
        f"Preview frames: {len(rows)} from distinct sessions, balanced across five mirrors. Accepted: {summary['accepted_frames']}; rejected: {summary['rejected_frames']}.\n\n"+
        "The RGB production detector, thresholds, 20% top expansion, <=10% outside-area rule, elapsed-time Kalman filtering, no alignment and bilinear resize are reused unchanged. IR is detected independently; RGB boxes are not transferred.\n\n"+
        f"Validated {len(audit)} candidate sources; skipped {summary['skipped_invalid_sources']} unreadable/inconsistent sources. Skips are listed in tables/source_selection_audit.csv. No blind JPEG/timestamp alignment was attempted.\n\n"+
        "This is a fixed-time preview, not a full-video retention estimate. Rejected samples are deliberately included. All accepted 224 PNG crops passed pixel-exact reload checks; raw video and IR timestamp SHA256 hashes were unchanged. No full production IR videos were created.\n\n"+
        "Review index.html or figures/contact_sheet_01.png through contact_sheet_05.png; full-resolution comparisons are in figures/frames/. Clinical absolute IR timestamps have not been assigned.\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Gallery: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
