"""Produce accepted-frame-only Kalman crops, without alignment, per raw session."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import av
import cv2
import numpy as np
import pandas as pd

from study.common.time_alignment import read_session_metadata
from .face_quality import FaceQualityGate, QualityConfig
from .mediapipe_alignment import ComparisonConfig, KalmanComparison
from .pipeline import CropConfig, MediaPipeDetector, RawRecording, VideoWriter
from .run_comparison import sha256


class InvalidSourceError(ValueError):
    """Raw JPEG packets cannot be safely mapped to recorder timestamps."""


def output_prefix(record):
    return "face224_ir" if record.get("modality") == "ir" else "face224"


def summary_directory(output, modality):
    return output / ("_face224_ir_processing" if modality == "ir" else "_face224_processing")


def source_signature(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_video(path, mapping, expected_digest):
    digest = hashlib.sha256()
    count = 0
    maximum_error = 0.
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if count >= len(mapping) or (frame.width, frame.height) != (224, 224):
                raise ValueError("Output dimensions/frame count mismatch")
            if not frame.key_frame:
                raise ValueError("FFV1 output must permit independent-frame decoding")
            error = abs(float(frame.time) - mapping[count]["source_elapsed_seconds"])
            maximum_error = max(maximum_error, error)
            if error > .00051:
                raise ValueError("Output presentation timestamp mismatch")
            mapping[count]["encoded_pts"] = frame.pts
            mapping[count]["encoded_time_base"] = str(frame.time_base)
            digest.update(frame.to_ndarray(format="bgr24").tobytes())
            count += 1
    if count != len(mapping) or digest.hexdigest() != expected_digest:
        raise ValueError("Decoded output differs from cropped pixels (lossless check)")
    return {"decoded_frames": count, "pixel_sha256": digest.hexdigest(),
            "pixel_exact": True, "all_frames_keyframes": True,
            "maximum_pts_error_seconds": maximum_error}


def process(record, output_path, protocol):
    cv2.setNumThreads(1)
    output = Path(output_path)
    directory = output / record["relative_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < 5 * 1024 ** 3:
        raise RuntimeError("Less than 5 GiB free")
    prefix = output_prefix(record)
    temporary = directory / f"{prefix}.partial.mkv"
    source = Path(record["raw_video_path"])
    source_paths = [source, Path(str(source) + ".ts")]
    try:
        before = {str(path): source_signature(path) for path in source_paths}
        hashes = {str(path): sha256(path) for path in source_paths}
        recording = RawRecording(source)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise InvalidSourceError(str(error)) from error
    detector = gate = writer = None
    mapping, audit, reasons = [], [], Counter()
    digest = hashlib.sha256()
    start = time.monotonic()
    print(f"[start] {record['video_id']} source_frames={len(recording.timestamps)}", flush=True)
    try:
        # Keep the reporting floor used in the pilot so association behavior does
        # not change just because the quality acceptance threshold is lowered.
        detector = MediaPipeDetector(CropConfig(confidence=.10))
        gate = FaceQualityGate(QualityConfig(**protocol["quality"]))
        processor = KalmanComparison(ComparisonConfig(**protocol["geometry"]), quality_gate=gate)
        for index, timestamp in enumerate(recording.timestamps):
            elapsed = float(timestamp - recording.timestamps[0])
            frame = recording.frame(index)
            if frame is None:
                row = {"direct_eligible": False, "rejection_reason": "source_decode_failed"}
            else:
                crops, row = processor.process(frame, *detector.detect_with_keypoints(frame), elapsed)
                if row["direct_eligible"]:
                    reason = "accepted"
                elif not row.get("quality_accepted", False):
                    reason = row.get("quality_reason", row["status"])
                else:
                    reason = row["status"]
                row["rejection_reason"] = reason
            output_index = len(mapping) if row["direct_eligible"] else -1
            times = {"source_frame_index": index, "source_elapsed_seconds": elapsed,
                     "source_recorder_timestamp_unix": float(timestamp),
                     "canonical_session_frame_time_unix":
                         record["session_time_unix"] + elapsed
                         if record["session_time_unix"] is not None and record.get("modality") != "ir" else np.nan}
            row.update({**times, "output_frame_index": output_index})
            audit.append(row)
            reasons[row["rejection_reason"]] += 1
            if row["direct_eligible"]:
                if writer is None:
                    writer = VideoWriter(temporary, 224, 224, lossless=True)
                    writer.stream.codec_context.gop_size = 1
                writer.write(crops["direct"], elapsed)
                digest.update(crops["direct"].tobytes())
                mapping.append({"output_frame_index": output_index, **times, "confidence": row["confidence"]})
        if writer is not None:
            writer.close()
            writer = None
        verification = verify_video(temporary, mapping, digest.hexdigest()) if mapping else None
        if {str(path): source_signature(path) for path in source_paths} != before:
            raise RuntimeError("Raw source changed during processing")
        pd.DataFrame(audit).to_csv(directory / f"{prefix}_audit.partial.csv", index=False)
        columns = ["output_frame_index", "source_frame_index", "source_elapsed_seconds",
                   "source_recorder_timestamp_unix", "canonical_session_frame_time_unix",
                   "confidence", "encoded_pts", "encoded_time_base"]
        pd.DataFrame(mapping, columns=columns).to_csv(directory / f"{prefix}_frames.partial.csv", index=False)
        result = {**record, "status": "completed" if mapping else "no_valid_frames",
                  "source_frames": len(recording.timestamps), "retained_frames": len(mapping),
                  "video_path": str(directory / f"{prefix}.mkv") if mapping else "",
                  "frames_path": str(directory / f"{prefix}_frames.csv"),
                  "bytes": temporary.stat().st_size if mapping else 0,
                  "rejection_counts": dict(reasons), "source_signatures": before,
                  "source_sha256": hashes, "protocol": protocol, "verification": verification,
                  "seconds": time.monotonic() - start}
        if mapping:
            temporary.replace(directory / f"{prefix}.mkv")
        else:
            (directory / f"{prefix}.mkv").unlink(missing_ok=True)
        for name in (f"{prefix}_audit", f"{prefix}_frames"):
            (directory / f"{name}.partial.csv").replace(directory / f"{name}.csv")
        (directory / f"{prefix}_metadata.partial.json").write_text(json.dumps(result, indent=2) + "\n")
        (directory / f"{prefix}_metadata.partial.json").replace(directory / f"{prefix}_metadata.json")
        print(f"[done] {record['video_id']} retained={len(mapping)}/{len(audit)} "
              f"bytes={result['bytes']} lossless_verified={bool(verification)} seconds={result['seconds']:.1f}", flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        if detector is not None:
            detector.close()
        if gate is not None:
            gate.close()
        recording.close()
        for name in (f"{prefix}.partial.mkv", f"{prefix}_audit.partial.csv", f"{prefix}_frames.partial.csv", f"{prefix}_metadata.partial.json"):
            (directory / name).unlink(missing_ok=True)


def inventory(raw_root, modality="rgb"):
    records = []
    for directory in sorted(raw_root.glob("mirror*_data/patient_*")):
        if not directory.is_dir():
            continue
        source_name = "raw_ir_video.avi" if modality == "ir" else "raw_video.avi"
        source = directory / source_name
        record = {"video_id": f"{directory.parent.name.removesuffix('_data')}_{directory.name}",
                  "relative_directory": str(directory.relative_to(raw_root)), "raw_video_path": str(source),
                  "hospital_id": "", "session_time_unix": None, "metadata_error": "",
                  "status": "pending" if source.is_file() and source.stat().st_size else "missing_or_empty_video"}
        if modality == "ir":
            record["modality"] = "ir"
        try:
            metadata = read_session_metadata(directory / "patient_info.txt",
                                             expected_local_id=directory.name.removeprefix("patient_"))
            record.update(hospital_id=metadata["session_hospital_id"], session_time_unix=metadata["session_time_unix"])
        except Exception as error:
            record["metadata_error"] = str(error)
        records.append(record)
    if not records:
        raise ValueError("No raw session directories found")
    discovered = {Path(record["raw_video_path"]) for record in records}
    source_name = "raw_ir_video.avi" if modality == "ir" else "raw_video.avi"
    if any(path not in discovered for path in raw_root.rglob(source_name)):
        raise ValueError(f"Raw {modality.upper()} video exists outside recognized mirror/patient layout")
    return records


def write_summary(output, records, results, modality="rgb"):
    output = summary_directory(output, modality)
    merged = {record["video_id"]: dict(record) for record in records}
    for result in results:
        merged[result["video_id"]] = {key: value for key, value in result.items()
                                      if key not in ("protocol", "verification", "source_signatures", "source_sha256", "rejection_counts")}
    pd.DataFrame(merged.values()).to_csv(output / "index.csv", index=False)
    status = Counter(record["status"] for record in merged.values())
    report = {"sessions": len(records), "statuses": dict(status),
              "source_frames_processed": sum(row.get("source_frames", 0) for row in results),
              "retained_frames": sum(row.get("retained_frames", 0) for row in results),
              "video_bytes": sum(row.get("bytes", 0) for row in results),
              "updated_unix": time.time()}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/root/shared/HealthMirrorRawData"))
    parser.add_argument("--output-dir", type=Path, help="Defaults to the original raw-video root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--modality", choices=("rgb", "ir"), default="rgb")
    parser.add_argument("--video-id", help="Optional single-video check using the same production protocol")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated face224 files; never modify raw sources")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.raw_root
    if args.workers < 1:
        raise ValueError("Invalid worker count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = summary_directory(args.output_dir, args.modality)
    summary_dir.mkdir(parents=True, exist_ok=True)
    lock = (summary_dir / ".processing.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {"version": 2, "quality": asdict(QualityConfig(confidence_threshold=.75)),
                "geometry": asdict(ComparisonConfig(enable_alignment=False)), "detector_reporting_floor": .10,
                "codec": "FFV1 level 3, BGR0, intra-only, lossless MKV", "size": [224, 224],
                "frames": "fresh accepted Kalman direct crops only; no alignment, gap filling or temporal resampling",
                "timestamps": "source elapsed PTS (container ~1 ms); exact original and session-canonical times in face224_frames.csv",
                "raw_root": str(args.raw_root.resolve()),
                "implementation_sha256": {name: sha256(Path(__file__).parent / name)
                                          for name in ("pipeline.py", "mediapipe_alignment.py", "face_quality.py")}}
    if args.modality == "ir":
        protocol.update(modality="ir", output_prefix="face224_ir",
                        timestamps="IR source elapsed PTS; exact recorder times in face224_ir_frames.csv; no independent elapsed-origin RGB/IR matching",
                        clinical_timestamp_assignment=False,
                        clinical_time_note="IR canonical times are not assigned without a common calibrated origin; paired clinical time comes from the validated RGB mapping",
                        source_handling="original IR MJPEG; no photometric enhancement; strict JPEG/timestamp count equality",
                        detector_weight_sha256=sha256(Path(__file__).parent / "models/blaze_face_short_range.tflite"))
    protocol_path = summary_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol and not args.overwrite:
        raise ValueError("Existing output uses a different protocol; choose another output directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    records = inventory(args.raw_root, args.modality)
    if args.video_id:
        records = [record for record in records if record["video_id"] == args.video_id]
        if len(records) != 1:
            raise ValueError("Unknown or nonunique video ID")
    if args.overwrite:
        for record in records:
            directory = args.output_dir / record["relative_directory"]
            prefix = output_prefix(record)
            for name in (f"{prefix}.mkv", f"{prefix}_frames.csv", f"{prefix}_audit.csv", f"{prefix}_metadata.json",
                         f"{prefix}.partial.mkv", f"{prefix}_frames.partial.csv", f"{prefix}_audit.partial.csv",
                         f"{prefix}_metadata.partial.json"):
                (directory / name).unlink(missing_ok=True)
        print(f"[overwrite] cleared generated face224 results for {len(records)} sessions; raw files untouched", flush=True)
    results, pending = [], []
    for record in records:
        if record["status"] != "pending":
            continue
        metadata_path = args.output_dir / record["relative_directory"] / f"{output_prefix(record)}_metadata.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            intact = all(Path(path).exists() and source_signature(path) == signature
                         for path, signature in previous["source_signatures"].items())
            intact = intact and all(previous.get(key) == record.get(key)
                                     for key in ("hospital_id", "session_time_unix", "metadata_error"))
            if (previous["protocol"] == protocol and intact and Path(previous["frames_path"]).exists()
                    and (previous["status"] == "no_valid_frames" or
                         (Path(previous["video_path"]).is_file() and Path(previous["video_path"]).stat().st_size == previous["bytes"]))):
                results.append(previous)
                continue
            raise ValueError(f"Existing result/source mismatch: {record['video_id']}")
        pending.append(record)
    total = len(pending) + len(results)
    write_summary(args.output_dir, records, results, args.modality)
    print(f"[inventory] sessions={len(records)} raw_{args.modality}={len(pending) + len(results)} "
          f"resume_completed={len(results)} queued={len(pending)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {}
        for record in pending:
            if shutil.disk_usage(args.output_dir).free < 5 * 1024 ** 3:
                raise RuntimeError("Less than 5 GiB free: processing stopped")
            futures[pool.submit(process, record, str(args.output_dir), protocol)] = record
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                status = "invalid_source" if args.modality == "ir" and isinstance(error, InvalidSourceError) else "failed"
                result = {**record, "status": status, "error": str(error)}
                print(f"[failed] {record['video_id']}: {error}", flush=True)
            results.append(result)
            write_summary(args.output_dir, records, results, args.modality)
            print(f"[progress] finished={len(results)}/{total} "
                  f"remaining={sum(not f.done() for f in futures)}", flush=True)
    print(f"[finished] output={args.output_dir}", flush=True)
    if any(row["status"] == "failed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
