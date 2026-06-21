"""Shared loading and preprocessing for raw mirror ECG windows."""

import glob
import os

import numpy as np
import pandas as pd


def read_ecg_file(path):
    """Read headerless timestamp,ECG logs with a tolerant fallback."""
    try:
        values = pd.read_csv(
            path,
            header=None,
            usecols=[0, 1],
            names=["timestamp", "ecg"],
            dtype=np.float64,
            on_bad_lines="skip",
        )
    except (TypeError, ValueError):
        values = pd.read_csv(
            path,
            header=None,
            usecols=[0, 1],
            names=["timestamp", "ecg"],
            on_bad_lines="skip",
        )
        values["timestamp"] = pd.to_numeric(values["timestamp"], errors="coerce")
        values["ecg"] = pd.to_numeric(values["ecg"], errors="coerce")

    timestamps = values["timestamp"].to_numpy(dtype=np.float64)
    ecg = values["ecg"].to_numpy(dtype=np.float64)
    valid_time = np.isfinite(timestamps)
    timestamps, ecg = timestamps[valid_time], ecg[valid_time]
    if len(timestamps) < 2:
        return None, None

    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], ecg[order]


def read_clean_ecg_file(path):
    """Read Timestamp/ECG files from mirror*_auto_cleaned_sqi."""
    values = pd.read_csv(path, usecols=["Timestamp", "ECG"])
    timestamps = pd.to_numeric(values["Timestamp"], errors="coerce").to_numpy()
    ecg = pd.to_numeric(values["ECG"], errors="coerce").to_numpy()
    valid_time = np.isfinite(timestamps)
    timestamps, ecg = timestamps[valid_time], ecg[valid_time]
    if len(timestamps) < 2:
        return None, None
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], ecg[order]


def raw_diagnostics(ecg):
    """Return simple pre-normalization artifact indicators."""
    finite = ecg[np.isfinite(ecg)]
    missing_fraction = 1.0 - float(np.isfinite(ecg).mean())
    if len(finite) < 2:
        return {
            "missing_fraction": missing_fraction,
            "raw_std": 0.0,
            "robust_amplitude": 0.0,
            "flat_fraction": 1.0,
            "clipping_fraction": 1.0,
            "impulse_ratio": float("inf"),
            "artifact_burden": 1.0,
        }

    raw_std = float(np.std(finite))
    low, high = np.percentile(finite, [5.0, 95.0])
    robust_amplitude = float(high - low)
    scale = max(robust_amplitude, 1e-8)
    differences = np.abs(np.diff(finite))

    flat_tolerance = max(1e-10, 1e-6 * scale)
    flat_fraction = float(np.mean(differences <= flat_tolerance))
    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    edge_tolerance = max(1e-10, 1e-5 * scale)
    clipping_fraction = max(
        float(np.mean(np.abs(finite - minimum) <= edge_tolerance)),
        float(np.mean(np.abs(finite - maximum) <= edge_tolerance)),
    )

    diff_median = float(np.median(differences))
    diff_mad = float(np.median(np.abs(differences - diff_median)))
    diff_scale = max(1.4826 * diff_mad, 1e-8)
    impulse_ratio = float(np.max(differences) / diff_scale)

    burden_components = (
        np.clip(missing_fraction / 0.05, 0.0, 1.0),
        np.clip((flat_fraction - 0.05) / 0.45, 0.0, 1.0),
        np.clip((clipping_fraction - 0.10) / 0.40, 0.0, 1.0),
        np.clip((impulse_ratio - 25.0) / 75.0, 0.0, 1.0),
    )
    return {
        "missing_fraction": missing_fraction,
        "raw_std": raw_std,
        "robust_amplitude": robust_amplitude,
        "flat_fraction": flat_fraction,
        "clipping_fraction": clipping_fraction,
        "impulse_ratio": impulse_ratio,
        "artifact_burden": float(max(burden_components)),
    }


def prepare_model_input(timestamps, ecg, window_sec, target_length):
    """Interpolate an irregular raw window and apply the pretraining z-score."""
    target_times = timestamps[0] + np.arange(target_length) * (
        window_sec / target_length
    )
    finite = np.isfinite(ecg)
    if finite.sum() < 2:
        resampled = np.zeros(target_length, dtype=np.float64)
    else:
        resampled = np.interp(target_times, timestamps[finite], ecg[finite])

    mean = float(np.mean(resampled))
    std = float(np.std(resampled))
    normalized = resampled - mean if std <= 1e-8 else (resampled - mean) / std
    return normalized.astype(np.float32)


def extract_window_from_arrays(
    timestamps,
    ecg,
    start_time,
    window_sec=10.0,
    target_length=1024,
    source_name="ECG arrays",
):
    """Extract a validated model window from ECG arrays already in memory."""
    start = int(np.searchsorted(timestamps, float(start_time), side="left"))
    end = int(
        np.searchsorted(timestamps, float(start_time) + window_sec, side="right")
    )
    if end - start < 16:
        raise ValueError(f"Insufficient samples at {start_time} in {source_name}")

    segment_time = timestamps[start:end]
    segment_ecg = ecg[start:end]
    if segment_time[-1] - segment_time[0] < 0.90 * window_sec:
        raise ValueError(
            f"Incomplete {window_sec}s window at {start_time}: {source_name}"
        )

    positive_diffs = np.diff(segment_time)
    positive_diffs = positive_diffs[positive_diffs > 0]
    if not len(positive_diffs) or np.max(positive_diffs) > 0.50:
        raise ValueError(f"Discontinuous ECG window at {start_time}: {source_name}")

    model_input = prepare_model_input(
        segment_time, segment_ecg, window_sec, target_length
    )
    return {
        "timestamps": segment_time,
        "raw_ecg": segment_ecg,
        "model_input": model_input,
        "diagnostics": raw_diagnostics(segment_ecg),
        "source_sampling_rate_hz": float(
            (len(segment_time) - 1)
            / max(segment_time[-1] - segment_time[0], 1e-8)
        ),
    }


def extract_window(
    file_path,
    start_time,
    window_sec=10.0,
    target_length=1024,
    data_source="raw",
):
    """Load one exact raw or cleaned window for annotation/training."""
    if data_source not in {"raw", "clean"}:
        raise ValueError("data_source must be 'raw' or 'clean'")
    reader = read_clean_ecg_file if data_source == "clean" else read_ecg_file
    timestamps, ecg = reader(file_path)
    if timestamps is None:
        raise ValueError(f"Could not read ECG data: {file_path}")
    return extract_window_from_arrays(
        timestamps,
        ecg,
        start_time,
        window_sec=window_sec,
        target_length=target_length,
        source_name=file_path,
    )


def load_raw_windows(
    data_root,
    window_sec,
    target_length,
    windows_per_file,
    max_files=None,
):
    """Uniformly sample raw windows across mirror*_data patient files."""
    paths = sorted(
        glob.glob(os.path.join(data_root, "mirror*_data", "patient_*", "ecg_log.csv"))
    )
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(
            f"No mirror*_data/patient_*/ecg_log.csv under {data_root}"
        )

    model_inputs = []
    records = []
    files_with_windows = 0

    print(f"[Raw ECG] Reading {len(paths)} files...")
    for file_index, path in enumerate(paths):
        timestamps, ecg = read_ecg_file(path)
        if timestamps is None:
            continue
        if float(timestamps[-1] - timestamps[0]) < window_sec:
            continue

        start_times = np.unique(
            np.linspace(
                timestamps[0],
                timestamps[-1] - window_sec,
                num=windows_per_file,
            )
        )
        mirror = os.path.basename(os.path.dirname(os.path.dirname(path)))
        patient = os.path.basename(os.path.dirname(path))
        windows_before_file = len(records)

        for window_index, start_time in enumerate(start_times):
            try:
                window = extract_window_from_arrays(
                    timestamps,
                    ecg,
                    start_time,
                    window_sec=window_sec,
                    target_length=target_length,
                    source_name=path,
                )
            except ValueError:
                continue

            record = {
                "window_id": len(records),
                "mirror": mirror,
                "patient_id": patient,
                "file_path": path,
                "window_index": int(window_index),
                "start_time": float(window["timestamps"][0]),
                "source_sampling_rate_hz": window["source_sampling_rate_hz"],
            }
            record.update(window["diagnostics"])
            records.append(record)
            model_inputs.append(window["model_input"])

        if len(records) > windows_before_file:
            files_with_windows += 1
        if (file_index + 1) % 250 == 0:
            print(
                f"[Raw ECG] files={file_index + 1}/{len(paths)}, "
                f"windows={len(records)}"
            )

    if not model_inputs:
        raise RuntimeError("No valid raw ECG windows were found.")

    print(
        f"[Raw ECG] Loaded {len(model_inputs)} windows from "
        f"{files_with_windows} files; skipped files={len(paths) - files_with_windows}"
    )
    return np.stack(model_inputs)[:, None, :], pd.DataFrame(records)
