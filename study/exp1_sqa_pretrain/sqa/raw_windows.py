"""Shared loading and preprocessing for raw mirror ECG windows."""

import glob
import os

import numpy as np
import pandas as pd


RAW_ECG_POLARITY = -1.0
CLEAN_ECG_POLARITY = 1.0


def polarity_for_source(data_source):
    """Return the required ECG polarity for a public data-source name."""
    if data_source == "raw":
        return RAW_ECG_POLARITY
    if data_source == "clean":
        return CLEAN_ECG_POLARITY
    raise ValueError("data_source must be 'raw' or 'clean'")


def read_ecg_file(path):
    """Read on-disk raw ECG values without applying the analysis polarity."""
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
    """Read on-disk cleaned ECG values without changing their polarity."""
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
    ecg = np.asarray(ecg)
    finite_mask = np.isfinite(ecg)
    finite = ecg[finite_mask]
    missing_fraction = 1.0 if not ecg.size else 1.0 - float(finite_mask.mean())
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


def _corrupt_signal(timestamps, ecg, corruption_type, severity, seed):
    """Apply a deterministic corruption in the signal's native amplitude scale."""
    corruption_type = str(corruption_type or "none").strip().lower()
    if corruption_type in {"", "none", "nan"}:
        return np.asarray(ecg, dtype=np.float64).copy()
    if corruption_type not in {
        "gaussian", "high_frequency", "clipping", "baseline",
        "impulse", "dropout",
    }:
        raise ValueError(f"Unsupported corruption_type: {corruption_type}")
    severity = int(severity)
    if severity not in {0, 1, 2}:
        raise ValueError("corruption_severity must be 0, 1, or 2")

    output = np.asarray(ecg, dtype=np.float64).copy()
    finite = np.isfinite(output)
    if finite.sum() < 2:
        return output
    rng = np.random.default_rng(int(seed))
    center = float(np.median(output[finite]))
    low, high = np.percentile(output[finite], [5.0, 95.0])
    scale = max(float(high - low) / 3.29, float(np.std(output[finite])) * 0.25, 1e-8)
    relative_time = np.asarray(timestamps, dtype=np.float64) - float(timestamps[0])

    if corruption_type == "gaussian":
        sigma = (0.20, 0.45, 0.90)[severity] * scale
        output[finite] += rng.normal(0.0, sigma, size=int(finite.sum()))
    elif corruption_type == "high_frequency":
        amplitude = (0.20, 0.45, 0.90)[severity] * scale
        frequency = rng.uniform(30.0, 45.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        output[finite] += amplitude * np.sin(
            2.0 * np.pi * frequency * relative_time[finite] + phase
        )
    elif corruption_type == "clipping":
        threshold = (1.50, 0.85, 0.45)[severity] * scale
        output[finite] = np.clip(
            output[finite], center - threshold, center + threshold
        )
    elif corruption_type == "baseline":
        amplitude = (0.50, 1.00, 1.80)[severity] * scale
        frequency = rng.uniform(0.10, 0.50)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        output[finite] += amplitude * np.sin(
            2.0 * np.pi * frequency * relative_time[finite] + phase
        )
    elif corruption_type == "impulse":
        count = (1, 3, 6)[severity]
        locations = rng.choice(np.flatnonzero(finite), size=count, replace=False)
        amplitudes = (4.0, 7.0, 11.0)[severity] * scale
        output[locations] += rng.choice((-1.0, 1.0), size=count) * amplitudes
    elif corruption_type == "dropout":
        duration = (0.50, 1.50, 3.00)[severity]
        total_duration = max(float(relative_time[-1]), duration)
        start_time = rng.uniform(0.0, max(0.0, total_duration - duration))
        dropout = (relative_time >= start_time) & (relative_time <= start_time + duration)
        output[dropout & finite] = center
    return output


def extract_window_from_arrays(
    timestamps,
    ecg,
    start_time,
    window_sec=10.0,
    target_length=1024,
    source_name="ECG arrays",
    polarity=None,
    corruption_type="none",
    corruption_severity=0,
    corruption_seed=0,
):
    """Extract a window; callers must explicitly declare array polarity."""
    if polarity is None:
        raise ValueError("extract_window_from_arrays requires explicit polarity")
    start = int(np.searchsorted(timestamps, float(start_time), side="left"))
    end = int(
        np.searchsorted(timestamps, float(start_time) + window_sec, side="right")
    )
    if end - start < 16:
        raise ValueError(f"Insufficient samples at {start_time} in {source_name}")

    segment_time = timestamps[start:end]
    segment_ecg = np.asarray(ecg[start:end], dtype=np.float64) * float(polarity)
    segment_ecg = _corrupt_signal(
        segment_time,
        segment_ecg,
        corruption_type,
        corruption_severity,
        corruption_seed,
    )
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
    polarity=None,
    corruption_type="none",
    corruption_severity=0,
    corruption_seed=0,
):
    """Load one exact raw or cleaned window for annotation/training."""
    required_polarity = polarity_for_source(data_source)
    reader = read_clean_ecg_file if data_source == "clean" else read_ecg_file
    if polarity is None:
        polarity = required_polarity
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
        polarity=polarity,
        corruption_type=corruption_type,
        corruption_severity=corruption_severity,
        corruption_seed=corruption_seed,
    )


def load_raw_windows(
    data_root,
    window_sec,
    target_length,
    windows_per_file,
    max_files=None,
):
    """Uniformly sample mirror*_data windows with ECG polarity=-1."""
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
                    polarity=RAW_ECG_POLARITY,
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
                "data_source": "raw",
                "polarity": RAW_ECG_POLARITY,
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
