"""Score every complete, non-overlapping raw ECG window with human-tuned SQA models."""

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.sqa.model import load_sqa_checkpoint
from exp1_sqa_pretrain.sqa.raw_windows import (
    RAW_ECG_POLARITY,
    extract_window_from_arrays,
    raw_diagnostics,
    read_ecg_file,
)


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = "/root/shared/HealthMirrorDataset"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_ROOT, "auto_scored")
DEFAULT_TCN_CHECKPOINT = os.path.join(
    _PKG_DIR,
    "checkpoints",
    "exp1_sqa_human_tcn_window_L1024_cumulative_round02_polarityfix_v1_best.pt",
)
DEFAULT_RESNET_CHECKPOINT = os.path.join(
    _PKG_DIR,
    "checkpoints",
    "exp1_sqa_human_resnet_window_L1024_cumulative_round02_polarityfix_v1_best.pt",
)

WINDOW_SCHEMA = pa.schema([
    ("window_id", pa.string()),
    ("mirror", pa.string()),
    ("patient_id", pa.string()),
    ("patient_key", pa.string()),
    ("window_index", pa.int32()),
    ("source_path", pa.string()),
    ("source_relative_path", pa.string()),
    ("source_start_row", pa.int64()),
    ("source_end_row_exclusive", pa.int64()),
    ("source_sample_count", pa.int32()),
    ("nominal_start_time", pa.float64()),
    ("nominal_end_time", pa.float64()),
    ("actual_start_time", pa.float64()),
    ("actual_end_time", pa.float64()),
    ("source_sampling_rate_hz", pa.float64()),
    ("window_sec", pa.float32()),
    ("data_source", pa.string()),
    ("polarity", pa.float32()),
    ("session_timestamp", pa.string()),
    ("external_patient_id", pa.string()),
    ("patient_info_json", pa.string()),
    ("status", pa.string()),
    ("error", pa.string()),
    ("missing_fraction", pa.float32()),
    ("raw_std", pa.float32()),
    ("robust_amplitude", pa.float32()),
    ("flat_fraction", pa.float32()),
    ("clipping_fraction", pa.float32()),
    ("impulse_ratio", pa.float32()),
    ("artifact_burden", pa.float32()),
    ("tcn_p_qrs", pa.float32()),
    ("tcn_p_morph", pa.float32()),
    ("resnet_p_qrs", pa.float32()),
    ("resnet_p_morph", pa.float32()),
    ("mean_p_qrs", pa.float32()),
    ("mean_p_morph", pa.float32()),
    ("min_p_qrs", pa.float32()),
    ("min_p_morph", pa.float32()),
    ("tcn_qrs_pass", pa.bool_()),
    ("tcn_morph_pass", pa.bool_()),
    ("resnet_qrs_pass", pa.bool_()),
    ("resnet_morph_pass", pa.bool_()),
    ("consensus_usable", pa.bool_()),
    ("timestamps", pa.list_(pa.float64())),
    ("ecg", pa.list_(pa.float32())),
])

RECORDING_SCHEMA = pa.schema([
    ("mirror", pa.string()),
    ("patient_id", pa.string()),
    ("patient_key", pa.string()),
    ("patient_dir", pa.string()),
    ("source_path", pa.string()),
    ("source_relative_path", pa.string()),
    ("source_size_bytes", pa.int64()),
    ("source_mtime_ns", pa.int64()),
    ("source_row_count", pa.int64()),
    ("recording_start_time", pa.float64()),
    ("recording_end_time", pa.float64()),
    ("recording_duration_sec", pa.float64()),
    ("estimated_sampling_rate_hz", pa.float64()),
    ("complete_window_count", pa.int32()),
    ("scored_window_count", pa.int32()),
    ("invalid_window_count", pa.int32()),
    ("tail_duration_sec", pa.float64()),
    ("status", pa.string()),
    ("error", pa.string()),
    ("session_timestamp", pa.string()),
    ("external_patient_id", pa.string()),
    ("patient_info_json", pa.string()),
    ("patient_info_raw", pa.string()),
    ("patient_files_json", pa.string()),
])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload, path):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _write_parquet(rows, schema, path):
    temporary = path + ".tmp"
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
    )
    os.replace(temporary, path)


def _parse_patient_info(patient_dir):
    path = os.path.join(patient_dir, "patient_info.txt")
    output = {
        "session_timestamp": "",
        "external_patient_id": "",
        "patient_info_json": "{}",
        "patient_info_raw": "",
    }
    if not os.path.isfile(path):
        return output
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = file.read()
        output["patient_info_raw"] = raw
        fields = {}
        for line in raw.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        output["session_timestamp"] = fields.get("Session Timestamp", "")
        encoded = fields.get("Patient Info", "")
        if encoded:
            decoded = json.loads(encoded)
            if isinstance(decoded, str):
                decoded = json.loads(decoded)
            if isinstance(decoded, dict):
                output["external_patient_id"] = str(decoded.get("patient_id", ""))
                output["patient_info_json"] = json.dumps(
                    decoded, ensure_ascii=False, sort_keys=True
                )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return output


def _recording_base(path, data_root):
    patient_dir = os.path.dirname(path)
    mirror = os.path.basename(os.path.dirname(patient_dir))
    patient_id = os.path.basename(patient_dir)
    patient = _parse_patient_info(patient_dir)
    stat = os.stat(path)
    return {
        "mirror": mirror,
        "patient_id": patient_id,
        "patient_key": f"{mirror}/{patient_id}",
        "patient_dir": patient_dir,
        "source_path": os.path.abspath(path),
        "source_relative_path": os.path.relpath(path, data_root),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        **patient,
        "patient_files_json": json.dumps(sorted(os.listdir(patient_dir))),
    }


def _thresholds(checkpoint):
    values = checkpoint.get("thresholds") or {}
    return {
        "qrs": float(values.get("qrs", 0.5)),
        "morph": float(values.get("morph", 0.5)),
    }


def _load_round02_model(path, expected_architecture, device):
    model, checkpoint = load_sqa_checkpoint(
        path, map_location="cpu", freeze_encoder=True
    )
    if checkpoint.get("task") != "ecg_sqa_human":
        raise ValueError(f"Not a human-fine-tuned SQA checkpoint: {path}")
    if checkpoint.get("encoder_architecture") != expected_architecture:
        raise ValueError(f"Expected {expected_architecture} checkpoint: {path}")
    sources = checkpoint.get("annotation_sources", [])
    if not any("round02" in item.get("queue_path", "") for item in sources):
        raise ValueError(f"Checkpoint does not contain round02 annotations: {path}")
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def _predict(model, inputs, batch_size, device):
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        outputs.append(model.predict_proba(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _complete_window_count(timestamps, window_sec):
    positive = np.diff(timestamps)
    positive = positive[positive > 0]
    if len(timestamps) < 2 or not len(positive):
        return 0, 0.0
    sample_interval = float(np.median(positive))
    covered_duration = float(timestamps[-1] - timestamps[0] + sample_interval)
    return int(np.floor((covered_duration + 1e-9) / window_sec)), covered_duration


def _continuous_runs(timestamps, max_gap_sec):
    """Return half-open row ranges separated by implausibly large time gaps."""
    boundaries = np.flatnonzero(np.diff(timestamps) > max_gap_sec) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(timestamps)]))
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= 2
    ], int(len(boundaries))


def _window_rows(
    path,
    data_root,
    window_sec,
    target_length,
    max_gap_sec,
    max_windows_per_recording,
):
    base = _recording_base(path, data_root)
    timestamps, ecg = read_ecg_file(path)
    if timestamps is None:
        recording = {
            **base,
            "source_row_count": 0,
            "recording_start_time": None,
            "recording_end_time": None,
            "recording_duration_sec": 0.0,
            "estimated_sampling_rate_hz": None,
            "complete_window_count": 0,
            "scored_window_count": 0,
            "invalid_window_count": 0,
            "tail_duration_sec": 0.0,
            "status": "read_error",
            "error": "Could not read at least two timestamped ECG rows",
        }
        return [], recording

    runs, gap_count = _continuous_runs(timestamps, max_gap_sec)
    positive = np.diff(timestamps)
    sampling_diffs = positive[(positive > 0) & (positive <= max_gap_sec)]
    estimated_fs = (
        float(1.0 / np.median(sampling_diffs)) if len(sampling_diffs) else None
    )
    plans = []
    total_covered_duration = 0.0
    total_tail_duration = 0.0
    for run_index, (run_start, run_end) in enumerate(runs):
        run_timestamps = timestamps[run_start:run_end]
        count, covered_duration = _complete_window_count(
            run_timestamps, window_sec
        )
        total_covered_duration += covered_duration
        total_tail_duration += max(0.0, covered_duration - count * window_sec)
        if count:
            plans.append((run_index, run_start, run_end, count))

    total_count = int(sum(plan[3] for plan in plans))
    discontinuity_note = (
        f"timestamp_discontinuities={gap_count}; continuous_runs={len(runs)}"
        if gap_count else ""
    )
    if total_count > max_windows_per_recording:
        recording = {
            **base,
            "source_row_count": len(timestamps),
            "recording_start_time": float(timestamps[0]),
            "recording_end_time": float(timestamps[-1]),
            "recording_duration_sec": total_covered_duration,
            "estimated_sampling_rate_hz": estimated_fs,
            "complete_window_count": total_count,
            "scored_window_count": 0,
            "invalid_window_count": total_count,
            "tail_duration_sec": total_tail_duration,
            "status": "window_limit_exceeded",
            "error": (
                f"Computed {total_count} windows, limit is "
                f"{max_windows_per_recording}; {discontinuity_note}"
            ),
        }
        return [], recording

    rows = []
    for _, run_start, run_end, run_count in plans:
        run_timestamps = timestamps[run_start:run_end]
        recording_start = float(run_timestamps[0])
        previous_end = None
        for local_window_index in range(run_count):
            window_index = len(rows)
            nominal_start = recording_start + local_window_index * window_sec
            nominal_end = nominal_start + window_sec
            start = run_start + int(np.searchsorted(
                run_timestamps, nominal_start, side="left"
            ))
            end = run_start + int(np.searchsorted(
                run_timestamps, nominal_end, side="left"
            ))
            if previous_end is not None and start != previous_end:
                raise RuntimeError(f"Non-contiguous source slicing in {path}")
            if not (run_start <= start < end <= run_end):
                raise RuntimeError(f"Empty or out-of-run window in {path}")
            previous_end = end
            segment_time = timestamps[start:end]
            segment_ecg = np.asarray(ecg[start:end], dtype=np.float64)
            analysis_ecg = (segment_ecg * RAW_ECG_POLARITY).astype(np.float32)
            diagnostics = raw_diagnostics(analysis_ecg)
            status, error, model_input, source_fs = "scored", "", None, None
            try:
                window = extract_window_from_arrays(
                    segment_time,
                    segment_ecg,
                    nominal_start,
                    window_sec=window_sec,
                    target_length=target_length,
                    source_name=path,
                    polarity=RAW_ECG_POLARITY,
                )
                model_input = window["model_input"]
                source_fs = window["source_sampling_rate_hz"]
            except ValueError as exception:
                status, error = "invalid", str(exception)

            rows.append({
                "window_id": f"{base['patient_key']}/w{window_index:04d}",
                "mirror": base["mirror"],
                "patient_id": base["patient_id"],
                "patient_key": base["patient_key"],
                "window_index": window_index,
                "source_path": base["source_path"],
                "source_relative_path": base["source_relative_path"],
                "source_start_row": start,
                "source_end_row_exclusive": end,
                "source_sample_count": end - start,
                "nominal_start_time": nominal_start,
                "nominal_end_time": nominal_end,
                "actual_start_time": float(segment_time[0]),
                "actual_end_time": float(segment_time[-1]),
                "source_sampling_rate_hz": source_fs,
                "window_sec": window_sec,
                "data_source": "raw",
                "polarity": RAW_ECG_POLARITY,
                "session_timestamp": base["session_timestamp"],
                "external_patient_id": base["external_patient_id"],
                "patient_info_json": base["patient_info_json"],
                "status": status,
                "error": error,
                **diagnostics,
                "tcn_p_qrs": None,
                "tcn_p_morph": None,
                "resnet_p_qrs": None,
                "resnet_p_morph": None,
                "mean_p_qrs": None,
                "mean_p_morph": None,
                "min_p_qrs": None,
                "min_p_morph": None,
                "tcn_qrs_pass": None,
                "tcn_morph_pass": None,
                "resnet_qrs_pass": None,
                "resnet_morph_pass": None,
                "consensus_usable": None,
                "timestamps": segment_time,
                "ecg": analysis_ecg,
                "_model_input": model_input,
            })

    scored = sum(row["status"] == "scored" for row in rows)
    recording = {
        **base,
        "source_row_count": len(timestamps),
        "recording_start_time": float(timestamps[0]),
        "recording_end_time": float(timestamps[-1]),
        "recording_duration_sec": total_covered_duration,
        "estimated_sampling_rate_hz": estimated_fs,
        "complete_window_count": total_count,
        "scored_window_count": scored,
        "invalid_window_count": total_count - scored,
        "tail_duration_sec": total_tail_duration,
        "status": (
            "complete_with_discontinuities" if gap_count and total_count
            else "complete" if total_count
            else "too_short"
        ),
        "error": discontinuity_note or (
            "" if total_count else f"Recording shorter than {window_sec:g}s"
        ),
    }
    return rows, recording


def _score_rows(rows, models, thresholds, batch_size, device):
    indices = [index for index, row in enumerate(rows) if row["_model_input"] is not None]
    if not indices:
        for row in rows:
            row.pop("_model_input", None)
        return
    inputs = np.stack([rows[index]["_model_input"] for index in indices])[:, None, :]
    predictions = {
        name: _predict(model, inputs, batch_size, device)
        for name, model in models.items()
    }
    for prediction_index, row_index in enumerate(indices):
        row = rows[row_index]
        tcn = predictions["tcn"][prediction_index]
        resnet = predictions["resnet"][prediction_index]
        row.update({
            "tcn_p_qrs": float(tcn[0]),
            "tcn_p_morph": float(tcn[1]),
            "resnet_p_qrs": float(resnet[0]),
            "resnet_p_morph": float(resnet[1]),
            "mean_p_qrs": float(np.mean([tcn[0], resnet[0]])),
            "mean_p_morph": float(np.mean([tcn[1], resnet[1]])),
            "min_p_qrs": float(min(tcn[0], resnet[0])),
            "min_p_morph": float(min(tcn[1], resnet[1])),
            "tcn_qrs_pass": bool(tcn[0] >= thresholds["tcn"]["qrs"]),
            "tcn_morph_pass": bool(tcn[1] >= thresholds["tcn"]["morph"]),
            "resnet_qrs_pass": bool(resnet[0] >= thresholds["resnet"]["qrs"]),
            "resnet_morph_pass": bool(resnet[1] >= thresholds["resnet"]["morph"]),
        })
        row["consensus_usable"] = all([
            row["tcn_qrs_pass"],
            row["tcn_morph_pass"],
            row["resnet_qrs_pass"],
            row["resnet_morph_pass"],
        ])
    for row in rows:
        row.pop("_model_input", None)


def _part_matches(path, expected_sources):
    if not os.path.isfile(path):
        return False
    try:
        table = pq.read_table(path, columns=["source_path"])
        return table.column("source_path").to_pylist() == expected_sources
    except (OSError, pa.ArrowException):
        return False


def _configuration(args, checkpoint_metadata):
    checkpoints = {}
    for name, path, metadata in (
        ("tcn", args.tcn_checkpoint, checkpoint_metadata["tcn"]),
        ("resnet", args.resnet_checkpoint, checkpoint_metadata["resnet"]),
    ):
        checkpoints[name] = {
            "path": os.path.abspath(path),
            "sha256": _sha256(path),
            "epoch": metadata.get("epoch"),
            "thresholds": _thresholds(metadata),
            "training_tag": metadata.get("training_config", {}).get("checkpoint_tag"),
        }
    return {
        "format_version": 2,
        "data_root": os.path.abspath(args.data_root),
        "raw_glob": "mirror*_data/patient_*/ecg_log.csv",
        "window_sec": args.window_sec,
        "target_length": args.target_length,
        "window_policy": "contiguous_non_overlapping_half_open",
        "incomplete_tail_policy": "record_in_recordings_table_but_do_not_score",
        "stored_ecg_polarity": RAW_ECG_POLARITY,
        "timestamp_discontinuity_policy": "split_into_continuous_runs",
        "max_timestamp_gap_sec": args.max_timestamp_gap_sec,
        "max_windows_per_recording": args.max_windows_per_recording,
        "patients_per_part": args.patients_per_part,
        "checkpoints": checkpoints,
    }


def _config_hash(configuration):
    encoded = json.dumps(configuration, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_readme(output_dir):
    text = "# Automatically scored raw ECG\n\n"
    text += "`windows/` is a partitioned Parquet dataset with one row per complete, "
    text += "non-overlapping 10-second ECG window. `timestamps` and `ecg` are list "
    text += "columns; ECG is stored with polarity `-1`, matching SQA inference. "
    text += "Intervals are half-open `[nominal_start_time, nominal_end_time)`, and "
    text += "source row ranges are also half-open, so adjacent windows share no samples. "
    text += "Timestamp gaps larger than the configured limit split a recording into "
    text += "independent continuous runs; windows never cross those gaps.\n\n"
    text += "`recordings/` contains one row per source recording, including patient "
    text += "metadata, file inventory, processing status, window counts, and incomplete "
    text += "tail duration. Invalid complete windows are retained with null scores.\n\n"
    text += "Example: `pandas.read_parquet('.../auto_scored/windows', "
    text += "columns=['window_id', 'tcn_p_qrs', 'resnet_p_qrs'])`. "
    text += "`consensus_usable` applies the validation-selected thresholds from both "
    text += "checkpoints; raw scores should be used if another operating point is desired.\n"
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as file:
        file.write(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score all contiguous non-overlapping raw ECG windows"
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tcn-checkpoint", default=DEFAULT_TCN_CHECKPOINT)
    parser.add_argument("--resnet-checkpoint", default=DEFAULT_RESNET_CHECKPOINT)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--patients-per-part", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-timestamp-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-windows-per-recording", type=int, default=1000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window_sec <= 0 or args.target_length < 16:
        raise ValueError("Invalid window configuration")
    if args.patients_per_part < 1 or args.batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_timestamp_gap_sec <= 0 or args.max_windows_per_recording < 1:
        raise ValueError("Timestamp safety limits must be positive")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    models, metadata = {}, {}
    for name, path in (
        ("tcn", args.tcn_checkpoint),
        ("resnet", args.resnet_checkpoint),
    ):
        models[name], metadata[name] = _load_round02_model(path, name, device)
    target_lengths = {int(item["target_length"]) for item in metadata.values()}
    window_lengths = {float(item["window_sec"]) for item in metadata.values()}
    if target_lengths != {args.target_length} or window_lengths != {args.window_sec}:
        raise ValueError("Checkpoint and scoring window configurations do not match")

    configuration = _configuration(args, metadata)
    configuration_hash = _config_hash(configuration)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(manifest_path):
        existing = json.load(open(manifest_path, "r", encoding="utf-8"))
        if existing.get("configuration_sha256") != configuration_hash:
            raise RuntimeError("Existing auto_scored configuration does not match")
        if existing.get("status") == "complete" and os.path.isfile(
            os.path.join(output_dir, "_SUCCESS")
        ):
            print(f"[Done] Existing complete dataset: {output_dir}")
            return
    else:
        _atomic_json({
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "configuration_sha256": configuration_hash,
            "configuration": configuration,
        }, manifest_path)
    _write_readme(output_dir)

    paths = sorted(glob.glob(os.path.join(
        os.path.abspath(args.data_root), "mirror*_data", "patient_*", "ecg_log.csv"
    )))
    if args.max_files is not None:
        paths = paths[:args.max_files]
    if not paths:
        raise FileNotFoundError("No raw ECG files found")

    thresholds = {name: _thresholds(item) for name, item in metadata.items()}
    print(f"[Start] files={len(paths)} device={device} output={output_dir}")
    processed_parts = 0
    mirrors = sorted({os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths})
    for mirror in mirrors:
        mirror_paths = [
            path for path in paths
            if os.path.basename(os.path.dirname(os.path.dirname(path))) == mirror
        ]
        windows_dir = os.path.join(output_dir, "windows", mirror)
        recordings_dir = os.path.join(output_dir, "recordings", mirror)
        os.makedirs(windows_dir, exist_ok=True)
        os.makedirs(recordings_dir, exist_ok=True)
        for part_index, offset in enumerate(
            range(0, len(mirror_paths), args.patients_per_part)
        ):
            batch_paths = mirror_paths[offset:offset + args.patients_per_part]
            window_path = os.path.join(windows_dir, f"part-{part_index:05d}.parquet")
            recording_path = os.path.join(
                recordings_dir, f"part-{part_index:05d}.parquet"
            )
            expected_sources = [os.path.abspath(path) for path in batch_paths]
            if _part_matches(recording_path, expected_sources) and os.path.isfile(window_path):
                print(f"[Resume] {mirror} part={part_index:05d}")
                processed_parts += 1
                continue

            window_rows, recording_rows = [], []
            for path in batch_paths:
                try:
                    rows, recording = _window_rows(
                        path,
                        args.data_root,
                        args.window_sec,
                        args.target_length,
                        args.max_timestamp_gap_sec,
                        args.max_windows_per_recording,
                    )
                except Exception as exception:
                    base = _recording_base(path, args.data_root)
                    rows = []
                    recording = {
                        **base,
                        "source_row_count": 0,
                        "recording_start_time": None,
                        "recording_end_time": None,
                        "recording_duration_sec": 0.0,
                        "estimated_sampling_rate_hz": None,
                        "complete_window_count": 0,
                        "scored_window_count": 0,
                        "invalid_window_count": 0,
                        "tail_duration_sec": 0.0,
                        "status": "error",
                        "error": f"{type(exception).__name__}: {exception}",
                    }
                window_rows.extend(rows)
                recording_rows.append(recording)

            _score_rows(
                window_rows, models, thresholds, args.batch_size, device
            )
            _write_parquet(window_rows, WINDOW_SCHEMA, window_path)
            _write_parquet(recording_rows, RECORDING_SCHEMA, recording_path)
            processed_parts += 1
            scored = sum(row["status"] == "scored" for row in window_rows)
            print(
                f"[Write] {mirror} part={part_index:05d} "
                f"files={len(batch_paths)} windows={len(window_rows)} scored={scored}"
            )

    window_files = sorted(glob.glob(os.path.join(output_dir, "windows", "*", "*.parquet")))
    recording_files = sorted(glob.glob(os.path.join(output_dir, "recordings", "*", "*.parquet")))
    window_table = pa.concat_tables([
        pq.read_table(path, columns=["status", "consensus_usable"]) for path in window_files
    ])
    recording_table = pa.concat_tables([
        pq.read_table(path, columns=["status", "complete_window_count", "tail_duration_sec"])
        for path in recording_files
    ])
    statuses = window_table.column("status").to_pylist()
    consensus = window_table.column("consensus_usable").to_pylist()
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "status": "complete",
        "created_at": json.load(open(manifest_path, "r", encoding="utf-8"))["created_at"],
        "completed_at": completed_at,
        "configuration_sha256": configuration_hash,
        "configuration": configuration,
        "counts": {
            "source_files": recording_table.num_rows,
            "complete_windows": window_table.num_rows,
            "scored_windows": sum(value == "scored" for value in statuses),
            "invalid_windows": sum(value != "scored" for value in statuses),
            "consensus_usable_windows": sum(value is True for value in consensus),
        },
        "layout": {
            "windows": "windows/<mirror>/part-*.parquet",
            "recordings": "recordings/<mirror>/part-*.parquet",
        },
    }
    _atomic_json(manifest, manifest_path)
    with open(os.path.join(output_dir, "_SUCCESS"), "w", encoding="utf-8") as file:
        file.write(completed_at + "\n")
    print(f"[Done] {json.dumps(manifest['counts'])}")


if __name__ == "__main__":
    main()
