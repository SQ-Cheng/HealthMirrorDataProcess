"""Phase 2 Data Cleaning & Corrected Feature Extraction.

Key fixes over Phase 1:
  1. Hemoglobin: raw values are BOTH g/dL (<20) and g/L (>=20).
     Converter must normalize ALL to g/L by multiplying only g/dL values ×10.
  2. Validate other analyte units for inconsistencies.
  3. Updated clinical thresholds using corrected hemoglobin.
  4. Output corrected feature dataset for all Phase 2 experiments.
"""

import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = "/root/shared/HealthMirrorDataset"
LAB_CSV = os.path.join(ROOT_DIR, "merged_lab_tests.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "outputs", "phase2")
SEED = 20260703
ECG_LENGTH = 256
PLACEHOLDER_HOSPITAL_IDS = {"", "-1", "1111111111", "1234567891", "nan", "None"}


# ═══════════════════════════════════════════════════════════════════════
# Utility functions (same as Phase 1, with hemoglobin fix)
# ═══════════════════════════════════════════════════════════════════════

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
    """Convert glucose: mg/dL → mmol/L (÷18). Only values > 30 are mg/dL."""
    out = values.copy()
    mask_mgdl = out > 30  # mg/dL values are typically 70-200
    out[mask_mgdl] = out[mask_mgdl] / 18.0
    return out


def _hemoglobin_to_gl_fixed(values):
    """FIXED: Convert hemoglobin to g/L. Only multiply g/dL values (<20) ×10."""
    out = values.copy()
    mask_gdl = out < 20  # g/dL: typical 8-18
    out[mask_gdl] = out[mask_gdl] * 10.0
    # g/L values (20-200) stay as-is
    return out


# Analyte definitions
_ANALYTE_MAP = {
    "lactate": {"item_names": ["乳酸浓度"], "converter": None},
    "troponin": {"item_names": ["*肌钙蛋白Ⅰ(hsTnI)测定", "肌钙蛋白Ⅰ(hsTnI)测定"], "converter": None},
    "glucose": {"item_names": ["*葡萄糖(Glu)测定", "葡萄糖浓度"], "converter": _glucose_to_mmol},
    "hemoglobin": {"item_names": ["*血红蛋白", "血红蛋白", "总血红蛋白"], "converter": _hemoglobin_to_gl_fixed},
    "po2": {"item_names": ["氧分压", "患者体温下氧分压"], "converter": None},
    "pco2": {"item_names": ["二氧化碳分压", "患者体温下二氧化碳分压"], "converter": None},
}


def validate_units():
    """Validate all analyte units and report inconsistencies."""
    print("=" * 60)
    print("Unit Validation")
    print("=" * 60)

    df = pd.read_csv(LAB_CSV, dtype=str, keep_default_na=False)

    for analyte_key, info in _ANALYTE_MAP.items():
        subset = df[df["检验项名称"].isin(info["item_names"])].copy()
        if subset.empty:
            continue
        subset["value"] = _extract_numeric(subset["检验值(文本)"])
        vals = subset["value"].dropna()

        if info["converter"] is not None:
            vals_converted = info["converter"](vals.values.copy())
        else:
            vals_converted = vals.values

        print(f"\n{analyte_key}: {len(vals)} values")
        print(f"  Raw range: [{vals.min():.1f}, {vals.max():.1f}]")
        print(f"  Raw median: {vals.median():.1f}")
        print(f"  Converted range: [{np.nanmin(vals_converted):.1f}, "
              f"{np.nanmax(vals_converted):.1f}]")
        print(f"  Converted median: {np.nanmedian(vals_converted):.1f}")
        print(f"  Units seen: {subset['单位'].value_counts().to_dict()}")

        # Flag suspicious
        if analyte_key == "hemoglobin":
            n_gdl = (vals < 20).sum()
            n_gl = (vals >= 20).sum()
            print(f"  Unit split: {n_gdl} g/dL, {n_gl} g/L")
            print(f"  After fix: median = {np.nanmedian(vals_converted):.0f} g/L (expected ~120-160)")

        if analyte_key == "glucose":
            n_mmol = (vals < 30).sum()
            n_mgdl = (vals >= 30).sum()
            print(f"  Unit split: {n_mmol} mmol/L, {n_mgdl} mg/dL")

    return


# ═══════════════════════════════════════════════════════════════════════
# ECG Feature Extraction (same as Phase 1)
# ═══════════════════════════════════════════════════════════════════════

def _compute_hr_from_ecg(ecg_signal, fs=25.6):
    n = len(ecg_signal)
    autocorr = np.correlate(ecg_signal, ecg_signal, mode="full")
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-12)

    min_lag = max(int(fs * 60 / 180), 2)
    max_lag = min(int(fs * 60 / 40), n // 2)
    if max_lag <= min_lag:
        return np.nan

    search = autocorr[min_lag:max_lag + 1]
    peak_lag = min_lag + np.argmax(search)
    if autocorr[peak_lag] < 0.3:
        return np.nan
    hr = 60.0 * fs / peak_lag
    return hr if 30 < hr < 200 else np.nan


def _compute_hrv_features(ecg_signal, fs=25.6):
    n = len(ecg_signal)
    threshold = 0.5 * np.max(np.abs(ecg_signal))
    diff_sign = np.diff(np.sign(np.diff(ecg_signal)))
    peak_indices = np.where(diff_sign < 0)[0] + 1
    peak_indices = peak_indices[ecg_signal[peak_indices] > threshold]
    if len(peak_indices) < 2:
        return {"sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan, "n_rr": 0}

    rr = np.diff(peak_indices) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 2:
        return {"sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan, "n_rr": len(rr)}

    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / max(len(np.diff(rr)), 1) * 100 if len(rr) > 1 else np.nan
    return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50, "n_rr": len(rr)}


def _compute_morphological_features(ecg_signal):
    return {
        "ecg_max": float(np.max(ecg_signal)),
        "ecg_min": float(np.min(ecg_signal)),
        "ecg_ptp": float(np.ptp(ecg_signal)),
        "ecg_rms": float(np.sqrt(np.mean(ecg_signal ** 2))),
        "ecg_skewness": float(skew(ecg_signal)),
        "ecg_kurtosis": float(kurtosis(ecg_signal)),
        "ecg_zero_crossing_rate": float(np.sum(np.diff(np.signbit(ecg_signal))) / len(ecg_signal)),
        "ecg_mean_abs": float(np.mean(np.abs(ecg_signal))),
    }


def _compute_spectral_features(ecg_signal, fs=25.6):
    n = len(ecg_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(ecg_signal)) ** 2
    total_power = np.sum(power)

    features = {"ecg_total_power": float(np.log1p(total_power))}
    bands = {"vlf": (0.003, 0.04), "lf": (0.04, 0.15), "hf": (0.15, 0.4)}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = np.sum(power[mask])
        features[f"ecg_{band_name}_power"] = float(np.log1p(bp))
        features[f"ecg_{band_name}_norm"] = float(bp / max(total_power, 1e-12))

    lf_p = np.sum(power[(freqs >= 0.04) & (freqs <= 0.15)])
    hf_p = np.sum(power[(freqs >= 0.15) & (freqs <= 0.4)])
    features["ecg_lf_hf_ratio"] = float(lf_p / max(hf_p, 1e-12))
    features["ecg_spectral_centroid"] = float(np.sum(freqs * power) / max(total_power, 1e-12))
    return features


def extract_all_ecg_features(ecg_signal, fs=25.6):
    features = {}
    features["hr"] = _compute_hr_from_ecg(ecg_signal, fs)
    features.update(_compute_hrv_features(ecg_signal, fs))
    features.update(_compute_morphological_features(ecg_signal))
    features.update(_compute_spectral_features(ecg_signal, fs))
    return features


# ═══════════════════════════════════════════════════════════════════════
# Signal I/O
# ═══════════════════════════════════════════════════════════════════════

def _read_cleaned_info():
    lookup = {}
    for info_path in sorted(glob.glob(os.path.join(DATA_ROOT, "mirror*_auto_cleaned_sqi",
                                                    "cleaned_patient_info.csv"))):
        mirror = os.path.basename(os.path.dirname(info_path)).split("_")[0]
        info = pd.read_csv(info_path, dtype=str, keep_default_na=False)
        for _, row in info.iterrows():
            key = (mirror, int(row["Lab_Patient_ID"]))
            lookup[key] = row.to_dict()
    return lookup


def _get_session_timestamp(signal_path):
    df = pd.read_csv(signal_path, usecols=["Timestamp"])
    ts = pd.to_numeric(df["Timestamp"], errors="coerce").dropna().to_numpy(np.float64)
    return float(np.median(ts)) if len(ts) > 0 else None


def _load_ecg(signal_path, length=ECG_LENGTH, window_sec=10.0):
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
    vector = (vector - float(np.mean(vector))) / max(std, 1e-8)
    return vector.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Lab timeseries builder (with surgery dates)
# ═══════════════════════════════════════════════════════════════════════

def build_lab_timeseries(lab_csv=LAB_CSV):
    """Build flat timeseries of lab measurements with corrected values."""
    df = pd.read_csv(lab_csv, dtype=str, keep_default_na=False)
    df["hospital_id"] = df["首页病案号"].apply(_normalize_hospital_id)
    df = df[df["hospital_id"] != ""].copy()
    df["timestamp_unix"] = _parse_datetime_to_unix(df["报告时间"])
    df = df.dropna(subset=["timestamp_unix"]).copy()

    # Extract patient demographics
    patient_info = {}
    for hid, group in df.groupby("hospital_id"):
        patient_info[hid] = {
            "sex": group["首页性别"].iloc[0] if len(group) > 0 else "unknown",
            "age": _extract_numeric(pd.Series([group["首页就诊时年龄"].iloc[0]])).iloc[0]
                   if len(group) > 0 else np.nan,
            "surgery_start": _parse_surgery_date(group["手术开始日期"].iloc[0])
                             if len(group) > 0 else None,
        }

    rows = []
    for analyte_key, info in _ANALYTE_MAP.items():
        subset = df[df["检验项名称"].isin(info["item_names"])].copy()
        if subset.empty:
            continue
        subset["value"] = _extract_numeric(subset["检验值(文本)"])
        if info["converter"] is not None:
            subset["value"] = info["converter"](subset["value"].values)
        subset = subset.dropna(subset=["value"]).copy()
        for _, row in subset.iterrows():
            rows.append({
                "hospital_id": row["hospital_id"],
                "analyte": analyte_key,
                "value": float(row["value"]),
                "timestamp_unix": int(row["timestamp_unix"]),
            })
    return pd.DataFrame(rows), patient_info


def _parse_surgery_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() in ("-", "", "nan", "None"):
        return None
    first = str(date_str).split("^")[0].strip()
    try:
        return pd.to_datetime(first)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Build corrected feature dataset
# ═══════════════════════════════════════════════════════════════════════

def build_corrected_dataset(output_dir=None, save=True):
    """Build the corrected Phase 2 feature dataset.

    Key additions over Phase 1:
      - Fixed hemoglobin conversion
      - Added patient demographics (sex, age)
      - Added surgery_date for trajectory modeling
      - Added days_from_surgery
      - Added previous_lab_value (most recent prior measurement)
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "direction_a")
    os.makedirs(output_dir, exist_ok=True)

    print("Building corrected lab timeseries...")
    lab_ts, patient_info = build_lab_timeseries()
    print(f"  Lab timeseries: {len(lab_ts)} measurements")
    print(f"  Patient info: {len(patient_info)} patients")

    print("Reading cleaned patient info...")
    info_lookup = _read_cleaned_info()

    signal_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "mirror*_auto_cleaned_sqi",
                                                  "patient_*.csv")))
    print(f"  Found {len(signal_paths)} signal files")

    # Pre-group lab by hospital_id
    lab_by_hospital = {}
    for hid in lab_ts["hospital_id"].unique():
        lab_by_hospital[hid] = lab_ts[lab_ts["hospital_id"] == hid]
    lab_by_hospital_sorted = {hid: grp.sort_values("timestamp_unix")
                              for hid, grp in lab_by_hospital.items()}

    rows = []
    n_matched = 0

    for sp in signal_paths:
        name = os.path.basename(sp)
        match = re.match(r"patient_(\d+)_(\d+)\.csv$", name)
        if not match:
            continue
        lab_patient_id = int(match.group(1))
        session_id = int(match.group(2))
        mirror_dir = os.path.basename(os.path.dirname(sp))
        mirror = mirror_dir.split("_")[0]

        info = info_lookup.get((mirror, lab_patient_id), {})
        hospital_id = str(info.get("Hospital_Patient_ID", ""))
        if not hospital_id or hospital_id in PLACEHOLDER_HOSPITAL_IDS:
            continue

        try:
            session_time = _get_session_timestamp(sp)
            if session_time is None:
                continue
        except Exception:
            continue

        try:
            ecg_signal = _load_ecg(sp)
            features = extract_all_ecg_features(ecg_signal)
        except Exception:
            continue

        # Build row
        row = {
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "session_id": session_id,
            "hospital_id": hospital_id,
            "sample_id": f"{mirror}_patient_{lab_patient_id:06d}_{session_id}",
            "capture_time_unix": session_time,
        }
        row.update(features)

        # Add patient demographics
        pinfo = patient_info.get(hospital_id, {})
        row["sex"] = pinfo.get("sex", "unknown")
        row["age"] = pinfo.get("age", np.nan)
        surgery_dt = pinfo.get("surgery_start", None)

        if surgery_dt is not None:
            row["days_from_surgery"] = (pd.to_datetime(session_time, unit="s") - surgery_dt).total_seconds() / 86400.0
        else:
            row["days_from_surgery"] = np.nan

        # Match lab values (time-closest) + PREVIOUS value
        lab_subset = lab_by_hospital.get(hospital_id)
        for analyte_key in ["lactate", "troponin", "glucose", "hemoglobin", "po2", "pco2"]:
            if lab_subset is not None:
                val, delta_h = _find_closest_measurement(lab_subset, hospital_id, session_time, analyte_key)
                prev_val = _find_previous_measurement(lab_by_hospital_sorted.get(hospital_id),
                                                       hospital_id, session_time, analyte_key)
                row[f"{analyte_key}_value"] = val
                row[f"{analyte_key}_delta_h"] = delta_h
                row[f"{analyte_key}_prev_value"] = prev_val
                if pd.notna(val) and pd.notna(prev_val):
                    row[f"{analyte_key}_delta_value"] = val - prev_val
                else:
                    row[f"{analyte_key}_delta_value"] = np.nan
            else:
                for suffix in ["_value", "_delta_h", "_prev_value", "_delta_value"]:
                    row[f"{analyte_key}{suffix}"] = np.nan

        rows.append(row)
        n_matched += 1

    print(f"  Matched: {n_matched}")
    features_df = pd.DataFrame(rows)

    # Compute corrected binary labels
    _compute_clinical_labels(features_df)

    if save:
        features_df.to_csv(os.path.join(output_dir, "ecg_features_corrected.csv"), index=False)
        print(f"Saved corrected dataset: {len(features_df)} samples, {features_df.shape[1]} columns")

    return features_df, patient_info


def _find_closest_measurement(lab_ts, hospital_id, session_time, analyte_key):
    subset = lab_ts[(lab_ts["analyte"] == analyte_key) &
                    (lab_ts["hospital_id"] == hospital_id)]
    if subset.empty:
        return np.nan, np.nan
    time_deltas = np.abs(subset["timestamp_unix"].to_numpy(np.float64) - session_time)
    best_idx = int(np.argmin(time_deltas))
    value = float(subset.iloc[best_idx]["value"])
    delta_h = float(time_deltas[best_idx]) / 3600.0
    return value, delta_h


def _find_previous_measurement(lab_ts_sorted, hospital_id, session_time, analyte_key):
    """Find the most recent lab measurement BEFORE session_time."""
    if lab_ts_sorted is None:
        return np.nan
    subset = lab_ts_sorted[(lab_ts_sorted["analyte"] == analyte_key) &
                           (lab_ts_sorted["hospital_id"] == hospital_id)]
    prev = subset[subset["timestamp_unix"] < session_time]
    if prev.empty:
        return np.nan
    return float(prev.iloc[-1]["value"])


def _compute_clinical_labels(df):
    """Compute corrected binary abnormality labels."""
    # Corrected hemoglobin thresholds (g/L):
    #   Male: < 130 g/L, Female: < 120 g/L
    df["hemoglobin_low"] = np.nan
    for idx, row in df.iterrows():
        hb = row["hemoglobin_value"]
        if pd.isna(hb):
            continue
        sex = str(row.get("sex", ""))
        thresh = 130.0 if sex == "男" else 120.0
        df.at[idx, "hemoglobin_low"] = int(hb < thresh)

    # Other thresholds (unchanged from Phase 1)
    df["lactate_abnormal"] = df["lactate_value"].apply(
        lambda x: np.nan if pd.isna(x) else int(x > 2.0))
    df["troponin_abnormal"] = df["troponin_value"].apply(
        lambda x: np.nan if pd.isna(x) else int(x > 34.0))
    df["glucose_abnormal"] = df["glucose_value"].apply(
        lambda x: np.nan if pd.isna(x) else int(x > 7.8))
    df["po2_abnormal"] = df["po2_value"].apply(
        lambda x: np.nan if pd.isna(x) else int(x < 80.0))
    df["pco2_abnormal"] = df["pco2_value"].apply(
        lambda x: np.nan if pd.isna(x) else int((x < 35.0) or (x > 45.0)))


if __name__ == "__main__":
    print("=== Phase 2: Data Cleaning & Validation ===\n")
    validate_units()
    print("\n\n=== Building Corrected Feature Dataset ===\n")
    df, pinfo = build_corrected_dataset()
    print(f"\nCorrected dataset: {len(df)} samples")
    print(f"\nHemoglobin value distribution (corrected):")
    hb_valid = df["hemoglobin_value"].dropna()
    print(f"  Count: {len(hb_valid)}")
    print(f"  Mean: {hb_valid.mean():.1f} g/L")
    print(f"  Median: {hb_valid.median():.1f} g/L")
    print(f"  Min: {hb_valid.min():.1f}, Max: {hb_valid.max():.1f}")
    print(f"\nLabel prevalence (corrected):")
    for c in ["lactate_abnormal", "troponin_abnormal", "glucose_abnormal",
              "hemoglobin_low", "po2_abnormal", "pco2_abnormal"]:
        valid = df[c].dropna()
        if len(valid) > 0:
            print(f"  {c}: {valid.mean():.2%} ({int(valid.sum())}/{len(valid)})")
