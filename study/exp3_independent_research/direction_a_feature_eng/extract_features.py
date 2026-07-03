"""Direction A: ECG-Derived Physiological Feature Engineering.

Extract classical ECG features (HR, HRV, morphological features) from raw ECG
signals and use them to predict lab test abnormalities. Compare with Exp2 DL.

Features extracted:
  - Heart Rate (HR)
  - HRV: SDNN, RMSSD, pNN50
  - Morphological: R-peak amplitude, QRS width, signal power in frequency bands
  - Statistical: mean, std, skewness, kurtosis, zero-crossing rate
  - Spectral: power in LF (0.04-0.15 Hz), HF (0.15-0.4 Hz) bands
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

from ..config import (
    ANALYTES,
    DATA_ROOT,
    ECG_LENGTH,
    LAB_CSV,
    OUTPUT_DIR,
    PLACEHOLDER_HOSPITAL_IDS,
    SEED,
)


# ──────────────────────────────────────────────────────────────────────
# Utility: load lab timeseries (reuse Exp2 logic)
# ──────────────────────────────────────────────────────────────────────

def _normalize_hospital_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    if text in PLACEHOLDER_HOSPITAL_IDS:
        return ""
    return text


def _extract_numeric(series):
    extracted = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _parse_datetime_to_unix(series):
    return pd.to_datetime(series, errors="coerce").astype("int64") // 10**9


def _glucose_to_mmol(values):
    return values / 18.0


def _hemoglobin_to_gl(values):
    return values * 10.0


_ANALYTE_MAP = {
    "lactate": {"item_names": ["乳酸浓度"], "converter": None},
    "troponin": {"item_names": ["*肌钙蛋白Ⅰ(hsTnI)测定", "肌钙蛋白Ⅰ(hsTnI)测定"], "converter": None},
    "glucose": {"item_names": ["*葡萄糖(Glu)测定", "葡萄糖浓度"], "converter": _glucose_to_mmol},
    "hemoglobin": {"item_names": ["*血红蛋白", "血红蛋白", "总血红蛋白"], "converter": _hemoglobin_to_gl},
    "po2": {"item_names": ["氧分压", "患者体温下氧分压"], "converter": None},
    "pco2": {"item_names": ["二氧化碳分压", "患者体温下二氧化碳分压"], "converter": None},
}


def build_lab_timeseries(lab_csv=LAB_CSV):
    """Build flat timeseries of lab measurements."""
    df = pd.read_csv(lab_csv, dtype=str, keep_default_na=False)
    df["hospital_id"] = df["首页病案号"].apply(_normalize_hospital_id)
    df = df[df["hospital_id"] != ""].copy()
    df["timestamp_unix"] = _parse_datetime_to_unix(df["报告时间"])
    df = df.dropna(subset=["timestamp_unix"]).copy()

    rows = []
    for analyte_key, info in _ANALYTE_MAP.items():
        subset = df[df["检验项名称"].isin(info["item_names"])].copy()
        if subset.empty:
            continue
        subset["value"] = _extract_numeric(subset["检验值(文本)"])
        if info["converter"] is not None:
            subset["value"] = info["converter"](subset["value"])
        subset = subset.dropna(subset=["value"]).copy()
        for _, row in subset.iterrows():
            rows.append({
                "hospital_id": row["hospital_id"],
                "analyte": analyte_key,
                "value": float(row["value"]),
                "timestamp_unix": int(row["timestamp_unix"]),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Signal loading
# ──────────────────────────────────────────────────────────────────────

def _read_cleaned_info(data_root=DATA_ROOT):
    """Build lookup: (mirror, lab_patient_id) -> patient info dict."""
    lookup = {}
    for info_path in sorted(
        glob.glob(os.path.join(data_root, "mirror*_auto_cleaned_sqi",
                               "cleaned_patient_info.csv"))
    ):
        mirror = os.path.basename(os.path.dirname(info_path)).split("_")[0]
        info = pd.read_csv(info_path, dtype=str, keep_default_na=False)
        for _, row in info.iterrows():
            key = (mirror, int(row["Lab_Patient_ID"]))
            lookup[key] = row.to_dict()
    return lookup


def _get_session_timestamp(signal_path):
    """Get median session timestamp (Unix epoch)."""
    df = pd.read_csv(signal_path, usecols=["Timestamp"])
    ts = pd.to_numeric(df["Timestamp"], errors="coerce").dropna().to_numpy(np.float64)
    if len(ts) == 0:
        return None
    return float(np.median(ts))


def _load_ecg(signal_path, length=ECG_LENGTH, window_sec=10.0):
    """Load and preprocess ECG: resample to fixed length, z-score normalize."""
    df = pd.read_csv(signal_path, usecols=["Timestamp", "ECG"])
    timestamps = pd.to_numeric(df["Timestamp"], errors="coerce").to_numpy(np.float64)
    ecg = pd.to_numeric(df["ECG"], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(timestamps) & np.isfinite(ecg)
    timestamps, ecg = timestamps[valid], ecg[valid]
    if len(timestamps) < 16:
        raise ValueError("Too few valid ECG samples")

    order = np.argsort(timestamps, kind="stable")
    timestamps, ecg = timestamps[order], ecg[order]

    duration = float(timestamps[-1] - timestamps[0])
    actual_window = min(window_sec, max(duration, 0.0))
    start_time = timestamps[0] + max(0.0, (duration - actual_window) / 2.0)
    target_times = start_time + np.linspace(0.0, actual_window, length, endpoint=False)
    vector = np.interp(target_times, timestamps, ecg)

    std = float(np.std(vector))
    if std <= 1e-8:
        vector = vector - float(np.mean(vector))
    else:
        vector = (vector - float(np.mean(vector))) / std
    return vector.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# ECG Feature Extraction
# ──────────────────────────────────────────────────────────────────────

def _compute_hr_from_ecg(ecg_signal, fs=25.6):
    """Estimate HR from ECG signal using autocorrelation-based period detection.

    Args:
        ecg_signal: normalized ECG of length ECG_LENGTH (256)
        fs: sampling frequency (256 samples / 10 sec = 25.6 Hz)

    Returns:
        hr: heart rate in bpm, or NaN if cannot determine
    """
    n = len(ecg_signal)
    # Autocorrelation
    autocorr = np.correlate(ecg_signal, ecg_signal, mode="full")
    autocorr = autocorr[n - 1:]  # positive lags only
    autocorr = autocorr / autocorr[0]  # normalize

    # Find peaks in reasonable HR range (40-180 bpm → 0.35-1.6 Hz → 16-73 samples lag)
    min_lag = int(fs * 60 / 180)   # 8.5 → 9
    max_lag = int(fs * 60 / 40)    # 38.4 → 38
    min_lag = max(min_lag, 2)
    max_lag = min(max_lag, n // 2)

    if max_lag <= min_lag:
        return np.nan

    search = autocorr[min_lag:max_lag + 1]
    if len(search) == 0:
        return np.nan

    peak_lag = min_lag + np.argmax(search)

    # Verify it's a meaningful peak (autocorr > 0.3)
    if autocorr[peak_lag] < 0.3:
        return np.nan

    hr = 60.0 * fs / peak_lag
    if hr < 30 or hr > 200:
        return np.nan
    return hr


def _compute_hrv_features(ecg_signal, fs=25.6):
    """Estimate HRV features from ECG signal.

    Simplified approach: use peak detection on the signal to find RR intervals.
    """
    n = len(ecg_signal)
    # Detect R-peaks using simple threshold
    threshold = 0.5 * np.max(np.abs(ecg_signal))
    # Find zero crossings of derivative as peak candidates
    diff = np.diff(np.sign(np.diff(ecg_signal)))
    peak_indices = np.where(diff < 0)[0] + 1
    # Filter by amplitude
    peak_indices = peak_indices[ecg_signal[peak_indices] > threshold]

    if len(peak_indices) < 2:
        return {"sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan, "n_rr": 0}

    rr_intervals = np.diff(peak_indices) / fs * 1000.0  # in ms
    # Filter unrealistic RR intervals (300-2000 ms)
    rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 2000)]

    if len(rr_intervals) < 2:
        return {"sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan, "n_rr": len(rr_intervals)}

    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else np.nan
    pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(np.diff(rr_intervals)) * 100 if len(rr_intervals) > 1 else np.nan

    return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50, "n_rr": len(rr_intervals)}


def _compute_morphological_features(ecg_signal):
    """Extract morphological features from ECG waveform."""
    features = {}
    # Amplitude features
    features["ecg_max"] = float(np.max(ecg_signal))
    features["ecg_min"] = float(np.min(ecg_signal))
    features["ecg_ptp"] = float(np.ptp(ecg_signal))  # peak-to-peak
    features["ecg_rms"] = float(np.sqrt(np.mean(ecg_signal ** 2)))

    # Shape features
    features["ecg_skewness"] = float(skew(ecg_signal))
    features["ecg_kurtosis"] = float(kurtosis(ecg_signal))
    features["ecg_zero_crossing_rate"] = float(
        np.sum(np.diff(np.signbit(ecg_signal))) / len(ecg_signal)
    )
    features["ecg_mean_abs"] = float(np.mean(np.abs(ecg_signal)))

    return features


def _compute_spectral_features(ecg_signal, fs=25.6):
    """Compute spectral power features."""
    n = len(ecg_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(ecg_signal)) ** 2

    # Total power
    total_power = np.sum(power)
    features = {"ecg_total_power": float(np.log1p(total_power))}

    # Band powers
    bands = {
        "vlf": (0.003, 0.04),
        "lf": (0.04, 0.15),
        "hf": (0.15, 0.4),
    }
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_power = np.sum(power[mask])
        features[f"ecg_{band_name}_power"] = float(np.log1p(band_power))
        if total_power > 0:
            features[f"ecg_{band_name}_norm"] = float(band_power / total_power)
        else:
            features[f"ecg_{band_name}_norm"] = 0.0

    # LF/HF ratio
    lf_power = np.sum(power[(freqs >= 0.04) & (freqs <= 0.15)])
    hf_power = np.sum(power[(freqs >= 0.15) & (freqs <= 0.4)])
    features["ecg_lf_hf_ratio"] = float(lf_power / max(hf_power, 1e-12))

    # Spectral centroid
    if total_power > 0:
        features["ecg_spectral_centroid"] = float(np.sum(freqs * power) / total_power)
    else:
        features["ecg_spectral_centroid"] = 0.0

    return features


def extract_all_ecg_features(ecg_signal, fs=25.6):
    """Extract all ECG features from a normalized signal.

    Returns:
        dict of feature_name -> float value
    """
    features = {}

    # HR
    features["hr"] = _compute_hr_from_ecg(ecg_signal, fs)

    # HRV
    hrv = _compute_hrv_features(ecg_signal, fs)
    features.update(hrv)

    # Morphological
    morph = _compute_morphological_features(ecg_signal)
    features.update(morph)

    # Spectral
    spec = _compute_spectral_features(ecg_signal, fs)
    features.update(spec)

    return features


# ──────────────────────────────────────────────────────────────────────
# Feature Extraction Pipeline
# ──────────────────────────────────────────────────────────────────────

def _find_closest_measurement(lab_ts, hospital_id, session_time, analyte_key):
    """Find lab measurement closest in time to session."""
    subset = lab_ts[(lab_ts["analyte"] == analyte_key) &
                    (lab_ts["hospital_id"] == hospital_id)]
    if subset.empty:
        return np.nan, np.nan

    time_deltas = np.abs(subset["timestamp_unix"].to_numpy(np.float64) - session_time)
    best_idx = int(np.argmin(time_deltas))
    value = float(subset.iloc[best_idx]["value"])
    delta_h = float(time_deltas[best_idx]) / 3600.0
    return value, delta_h


def build_feature_dataset(output_dir=None, save=True):
    """Extract ECG features and match with lab values.

    Returns:
        features_df: DataFrame with ECG features + lab values
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "direction_a")
    os.makedirs(output_dir, exist_ok=True)

    print("Building lab timeseries...")
    lab_ts = build_lab_timeseries()
    print(f"  Lab timeseries: {len(lab_ts)} measurements")

    print("Reading cleaned patient info...")
    info_lookup = _read_cleaned_info()

    print("Scanning signal files...")
    signal_paths = sorted(
        glob.glob(os.path.join(DATA_ROOT, "mirror*_auto_cleaned_sqi",
                               "patient_*.csv"))
    )
    print(f"  Found {len(signal_paths)} signal files")

    # Pre-group lab timeseries by hospital_id
    lab_by_hospital = {}
    for hid in lab_ts["hospital_id"].unique():
        lab_by_hospital[hid] = lab_ts[lab_ts["hospital_id"] == hid]

    rows = []
    n_failures = 0
    n_matched = 0

    for sp in signal_paths:
        # Parse signal path
        name = os.path.basename(sp)
        match = re.match(r"patient_(\d+)_(\d+)\.csv$", name)
        if not match:
            continue
        lab_patient_id = int(match.group(1))
        session_id = int(match.group(2))

        # Get mirror from path
        mirror_dir = os.path.basename(os.path.dirname(sp))
        mirror = mirror_dir.split("_")[0]

        # Get patient info
        info = info_lookup.get((mirror, lab_patient_id), {})
        hospital_id = str(info.get("Hospital_Patient_ID", ""))
        if not hospital_id or hospital_id in PLACEHOLDER_HOSPITAL_IDS:
            continue

        # Get session time
        try:
            session_time = _get_session_timestamp(sp)
            if session_time is None:
                n_failures += 1
                continue
        except Exception:
            n_failures += 1
            continue

        # Load ECG
        try:
            ecg_signal = _load_ecg(sp)
        except Exception:
            n_failures += 1
            continue

        # Extract features
        features = extract_all_ecg_features(ecg_signal)

        # Match lab values
        row = {
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "session_id": session_id,
            "hospital_id": hospital_id,
            "sample_id": f"{mirror}_patient_{lab_patient_id:06d}_{session_id}",
            "capture_time_unix": session_time,
        }
        row.update(features)

        # Match each analyte
        lab_subset = lab_by_hospital.get(hospital_id)
        for analyte_key in ANALYTES:
            if lab_subset is not None:
                val, delta_h = _find_closest_measurement(
                    lab_subset, hospital_id, session_time, analyte_key)
                row[f"{analyte_key}_value"] = val
                row[f"{analyte_key}_delta_h"] = delta_h
            else:
                row[f"{analyte_key}_value"] = np.nan
                row[f"{analyte_key}_delta_h"] = np.nan

        rows.append(row)
        n_matched += 1

    print(f"  Matched: {n_matched}, Failures: {n_failures}")

    features_df = pd.DataFrame(rows)

    # Compute binary labels using clinical thresholds
    thresholds = {
        "lactate": (2.0, "gt"),
        "troponin": (34.0, "gt"),
        "glucose": (7.8, "gt"),
        "hemoglobin": (120.0, "lt"),  # female; male would be 130
        "po2": (80.0, "lt"),
        "pco2": (45.0, "gt"),  # high; also low < 35
    }

    for analyte, (thresh, op) in thresholds.items():
        val_col = f"{analyte}_value"
        label_col = f"{analyte}_abnormal"
        features_df[label_col] = features_df[val_col].apply(
            lambda x: np.nan if pd.isna(x) else
            (int(x > thresh) if op == "gt" else int(x < thresh))
        )

    # Also add sex-dependent hemoglobin
    # (We don't have sex in the feature rows easily, but we can estimate from info)
    # For now, use 125 as unisex threshold
    features_df["hemoglobin_low"] = features_df["hemoglobin_value"].apply(
        lambda x: np.nan if pd.isna(x) else int(x < 125.0))

    if save:
        features_df.to_csv(os.path.join(output_dir, "ecg_features.csv"), index=False)
        print(f"Saved to {output_dir}/ecg_features.csv")

    return features_df


if __name__ == "__main__":
    df = build_feature_dataset()
    print(f"\nFeature dataset: {len(df)} rows, {df.shape[1]} columns")
    print(f"Feature columns: {[c for c in df.columns if not c.endswith('_value') and not c.endswith('_delta_h') and not c.endswith('_abnormal') and c not in ('mirror','lab_patient_id','session_id','hospital_id','sample_id','capture_time_unix')]}")
    print(f"\nLabel prevalence:")
    for c in [c for c in df.columns if c.endswith("_abnormal") or c == "hemoglobin_low"]:
        valid = df[c].dropna()
        if len(valid) > 0:
            print(f"  {c}: {valid.mean():.2%} positive ({int(valid.sum())}/{len(valid)})")
