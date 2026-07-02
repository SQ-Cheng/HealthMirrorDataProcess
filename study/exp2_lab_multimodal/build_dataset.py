"""Dataset builder for Exp2: time-matched ECG/face features and lab-test labels.

Key improvement over v1:
    Labels are now TIME-MATCHED — for each ECG capture session, we find the
    temporally closest lab measurement for each analyte, rather than taking
    the patient-level max/min across the entire hospital stay.

Produces:
    outputs/manifest.csv   — sample-level metadata + binary labels
    outputs/features.npz   — {'sample_id', 'hospital_id', 'ecg', 'face', 'targets'}
    outputs/label_summary.csv — per-target statistics
"""

import argparse
import glob
import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    DATA_ROOT,
    ECG_LENGTH,
    ECG_WINDOW_SEC,
    FACE_FRAME_INDEX,
    FACE_SIZE,
    LAB_CSV,
    OUTPUT_DIR,
    PLACEHOLDER_HOSPITAL_IDS,
    SEED,
    TARGETS,
)


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════

def _ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def _normalize_hospital_id(value):
    """Normalize hospital patient ID string."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    if text in PLACEHOLDER_HOSPITAL_IDS:
        return ""
    return text


def _extract_numeric(series):
    """Extract first numeric value from a string series."""
    extracted = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _parse_datetime_to_unix(series):
    """Convert datetime strings like '2026-01-23 14:56:40' to Unix epoch seconds."""
    return pd.to_datetime(series, errors="coerce").astype("int64") // 10**9


# Unit converters
def _glucose_to_mmol(values):
    return values / 18.0


def _hemoglobin_to_gl(values):
    return values * 10.0


# ═══════════════════════════════════════════════════════════════════════
# Analyte definitions: (target_name, item_names, unit_converter, direction)
#   direction: 'max' = abnormality is high value; 'min' = low value
# ═══════════════════════════════════════════════════════════════════════

_ANALYTE_MAP = {
    "lactate": {
        "item_names": ["乳酸浓度"],
        "converter": None,
        "direction": "max",
    },
    "troponin": {
        "item_names": ["*肌钙蛋白Ⅰ(hsTnI)测定", "肌钙蛋白Ⅰ(hsTnI)测定"],
        "converter": None,
        "direction": "max",
    },
    "glucose": {
        "item_names": ["*葡萄糖(Glu)测定", "葡萄糖浓度"],
        "converter": _glucose_to_mmol,
        "direction": "max",
    },
    "hemoglobin": {
        "item_names": ["*血红蛋白", "血红蛋白", "总血红蛋白"],
        "converter": _hemoglobin_to_gl,
        "direction": "min",
    },
    "po2": {
        "item_names": ["氧分压", "患者体温下氧分压"],
        "converter": None,
        "direction": "min",
    },
    "pco2": {
        "item_names": ["二氧化碳分压", "患者体温下二氧化碳分压"],
        "converter": None,
        "direction": "max",  # both min and max used for different targets
    },
}

# Threshold rules: (target_name, analyte_key, threshold, op, sex_dependent)
_THRESHOLD_RULES = [
    # Standard thresholds
    ("lactate_high",           "lactate",     2.0,    "gt",   False),
    ("troponin_high",          "troponin",    34.0,   "gt",   False),
    ("glucose_high",           "glucose",     7.8,    "gt",   False),
    ("po2_low",                "po2",         80.0,   "lt",   False),
    ("pco2_low",               "pco2",        34.0,   "lt",   False),
    ("pco2_high",              "pco2",        50.0,   "gt",   False),
    # Severity thresholds
    ("lactate_moderate_high",  "lactate",     4.0,    "gt",   False),
    ("troponin_extreme_high",  "troponin",    1000.0, "gt",   False),
    ("glucose_marked_high",    "glucose",     10.0,   "gt",   False),
    ("hemoglobin_moderate_low","hemoglobin",  90.0,   "lt",   False),
    ("po2_moderate_low",       "po2",         70.0,   "lt",   False),
]


def _apply_threshold(value, threshold, op):
    """Apply comparison operator."""
    if np.isnan(value):
        return np.nan
    if op == "gt":
        return int(value > threshold)
    elif op == "lt":
        return int(value < threshold)
    return np.nan


# ═══════════════════════════════════════════════════════════════════════
# Lab timeseries builder
# ═══════════════════════════════════════════════════════════════════════

def _build_lab_timeseries(lab_csv):
    """Build a flat timeseries of individual lab measurements.

    Returns DataFrame with columns:
        hospital_id, analyte, value, timestamp_unix
    """
    df = pd.read_csv(lab_csv, dtype=str, keep_default_na=False)
    df["hospital_id"] = df["首页病案号"].apply(_normalize_hospital_id)
    df = df[df["hospital_id"] != ""].copy()

    # Parse timestamps
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


# ═══════════════════════════════════════════════════════════════════════
# Signal / frame readers
# ═══════════════════════════════════════════════════════════════════════

def _parse_signal_file(path):
    """Parse patient ID and session ID from signal filename."""
    name = os.path.basename(path)
    match = re.match(r"patient_(\d+)_(\d+)\.csv$", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_cleaned_info(data_root):
    """Build lookup: (mirror, lab_patient_id) → patient info dict."""
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
    """Get the capture timestamp (median) of an ECG session, as Unix epoch seconds."""
    df = pd.read_csv(signal_path, usecols=["Timestamp"])
    ts = pd.to_numeric(df["Timestamp"], errors="coerce").dropna().to_numpy(np.float64)
    if len(ts) == 0:
        return None
    return float(np.median(ts))


def _extract_mjpeg_frame(video_path, frame_index=FACE_FRAME_INDEX):
    """Extract a single JPEG frame from an MJPEG video without ffmpeg/cv2."""
    with open(video_path, "rb") as f:
        data = f.read(10 * 1024 * 1024)
    starts = []
    pos = 0
    marker = b"\xff\xd8"
    while True:
        pos = data.find(marker, pos)
        if pos < 0:
            break
        starts.append(pos)
        pos += 2
        if len(starts) > frame_index + 5:
            break
    if len(starts) <= frame_index:
        with open(video_path, "rb") as f:
            data = f.read()
        starts = []
        pos = 0
        while True:
            pos = data.find(marker, pos)
            if pos < 0:
                break
            starts.append(pos)
            pos += 2
    if not starts:
        raise ValueError("No JPEG SOI marker found in video")
    start = starts[min(frame_index, len(starts) - 1)]
    end = data.find(b"\xff\xd9", start + 2)
    if end < 0:
        raise ValueError("No JPEG EOI marker found in video")
    return Image.open(BytesIO(data[start:end + 2])).convert("RGB")


def _load_face(video_path, sample_id, output_dir, face_size):
    """Load/cache face frame, return normalized grayscale (32×32)."""
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    cache_path = os.path.join(frame_dir, f"{sample_id}.jpg")
    if os.path.exists(cache_path):
        image = Image.open(cache_path).convert("RGB")
    else:
        image = _extract_mjpeg_frame(video_path)
        image.save(cache_path, quality=90)
    small = image.resize((face_size, face_size), Image.BILINEAR)
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    return gray


def _load_ecg(signal_path, length, window_sec):
    """Load and preprocess ECG signal: resample to fixed length, z-score normalize."""
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

    # Z-score normalize
    std = float(np.std(vector))
    if std <= 1e-8:
        vector = vector - float(np.mean(vector))
    else:
        vector = (vector - float(np.mean(vector))) / std

    return vector.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Time-matched label computation
# ═══════════════════════════════════════════════════════════════════════

def _find_closest_measurement(lab_ts, session_time, analyte_key):
    """Find the lab measurement closest in time to the session.

    Args:
        lab_ts: DataFrame with columns [analyte, value, timestamp_unix]
        session_time: Unix epoch seconds of ECG capture

    Returns:
        (value, time_delta_hours) or (nan, nan) if no measurement found
    """
    subset = lab_ts[lab_ts["analyte"] == analyte_key]
    if subset.empty:
        return np.nan, np.nan

    time_deltas = np.abs(subset["timestamp_unix"].to_numpy(np.float64) - session_time)
    best_idx = int(np.argmin(time_deltas))
    value = float(subset.iloc[best_idx]["value"])
    delta_sec = float(time_deltas[best_idx])
    return value, delta_sec / 3600.0


def _compute_labels_for_session(lab_ts, session_time, sex_value, surgery_text):
    """Compute all binary labels for one ECG capture session using time-matched values.

    Returns:
        dict: {target_name: label (0/1/nan), ...}
        dict: {target_name: raw_value, ...}  (for debugging)
        dict: {target_name: time_delta_hours, ...}
    """
    labels = {}
    raw_values = {}
    time_deltas = {}

    # Collect time-matched values for each analyte
    analyte_values = {}
    for analyte_key in _ANALYTE_MAP:
        val, delta_h = _find_closest_measurement(lab_ts, session_time, analyte_key)
        analyte_values[analyte_key] = val
        raw_values[analyte_key] = val
        time_deltas[analyte_key] = delta_h

    # Apply threshold rules
    for target_name, analyte_key, threshold, op, sex_dep in _THRESHOLD_RULES:
        val = analyte_values.get(analyte_key, np.nan)
        labels[target_name] = _apply_threshold(val, threshold, op)

    # Hemoglobin_low: sex-dependent threshold
    hb_val = analyte_values.get("hemoglobin", np.nan)
    if sex_value == "男":
        hb_threshold = 130.0
    else:
        hb_threshold = 120.0
    labels["hemoglobin_low"] = _apply_threshold(hb_val, hb_threshold, "lt")

    # pco2_abnormal: both low AND high
    pco2_val = analyte_values.get("pco2", np.nan)
    if np.isnan(pco2_val):
        labels["pco2_abnormal"] = np.nan
    else:
        labels["pco2_abnormal"] = int((pco2_val < 35.0) or (pco2_val > 45.0))

    # coronary_context: from surgery text
    labels["coronary_context"] = int("冠心病" in str(surgery_text))

    # high_blood_pressure: computed per-sample from BP readings (handled separately)
    labels["high_blood_pressure"] = np.nan

    return labels, raw_values, time_deltas


# ═══════════════════════════════════════════════════════════════════════
# Main dataset builder
# ═══════════════════════════════════════════════════════════════════════

def _build_samples(data_root, lab_timeseries, output_dir, max_samples=None):
    """Iterate over signal files, extract ECG + face, compute time-matched labels.

    Returns:
        manifest: DataFrame with metadata and labels.
        ecg_array:  (N, ECG_LENGTH) float32
        face_array: (N, FACE_SIZE, FACE_SIZE) float32
    """
    info_lookup = _read_cleaned_info(data_root)

    # Pre-group lab timeseries by hospital_id for fast lookup
    lab_by_hospital = {}
    for hid, group in lab_timeseries.groupby("hospital_id"):
        lab_by_hospital[hid] = group

    manifest_rows = []
    ecg_list = []
    face_list = []
    failures = []

    signal_paths = sorted(
        glob.glob(os.path.join(data_root, "mirror*_auto_cleaned_sqi",
                               "patient_*.csv"))
    )
    rng = np.random.default_rng(SEED)
    if max_samples is not None and len(signal_paths) > max_samples:
        signal_paths = list(rng.choice(signal_paths, size=max_samples, replace=False))
        signal_paths.sort()

    for signal_path in signal_paths:
        parsed = _parse_signal_file(signal_path)
        if parsed is None:
            continue
        lab_patient_id, session_id = parsed
        mirror = os.path.basename(os.path.dirname(signal_path)).split("_")[0]

        info = info_lookup.get((mirror, lab_patient_id))
        if info is None:
            continue

        hospital_id = _normalize_hospital_id(info.get("Hospital_Patient_ID", ""))
        if hospital_id == "":
            continue

        # Get lab timeseries for this hospital
        patient_lab = lab_by_hospital.get(hospital_id)
        if patient_lab is None or patient_lab.empty:
            continue

        # Get session capture timestamp
        session_time = _get_session_timestamp(signal_path)
        if session_time is None:
            continue

        sample_id = f"{mirror}_patient_{lab_patient_id:06d}_{session_id}"
        video_path = os.path.join(
            data_root, f"{mirror}_data", f"patient_{lab_patient_id:06d}", "video.avi"
        )

        try:
            ecg_vec = _load_ecg(signal_path, ECG_LENGTH, ECG_WINDOW_SEC)
            face_mat = _load_face(video_path, sample_id, output_dir, FACE_SIZE)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            continue

        # Get sex and surgery text from lab data for this hospital
        sex_value = info.get("sex", "")
        # Get surgery text from the lab CSV for this hospital
        surgery_text = ""
        lab_df = pd.read_csv(LAB_CSV, dtype=str, keep_default_na=False)
        lab_df["_hid"] = lab_df["首页病案号"].apply(_normalize_hospital_id)
        hosp_rows = lab_df[lab_df["_hid"] == hospital_id]
        if not hosp_rows.empty:
            surgery_text = " ".join(hosp_rows["首页手术操作名称"].dropna().astype(str))

        # Compute time-matched labels
        labels, raw_vals, time_deltas = _compute_labels_for_session(
            patient_lab, session_time, sex_value, surgery_text
        )

        # Compute high_blood_pressure from patient info (session-level BP)
        low_bp = pd.to_numeric(info.get("Low_Blood_Pressure", -1), errors="coerce")
        high_bp = pd.to_numeric(info.get("High_Blood_Pressure", -1), errors="coerce")
        if pd.notna(high_bp) and pd.notna(low_bp) and high_bp > 0 and low_bp > 0:
            labels["high_blood_pressure"] = int(high_bp >= 140.0 or low_bp >= 90.0)

        row_data = {
            "sample_id": sample_id,
            "hospital_id": hospital_id,
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "session_id": session_id,
            "capture_time_unix": session_time,
        }
        row_data.update(labels)
        manifest_rows.append(row_data)
        ecg_list.append(ecg_vec)
        face_list.append(face_mat)

    if failures:
        print(f"  [WARN] {len(failures)} samples failed to load:")
        for f in failures[:5]:
            print(f"    {f['sample_id']}: {f['error']}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")

    manifest = pd.DataFrame(manifest_rows)
    ecg_array = np.stack(ecg_list, axis=0) if ecg_list else np.empty((0, ECG_LENGTH), dtype=np.float32)
    face_array = np.stack(face_list, axis=0) if face_list else np.empty((0, FACE_SIZE, FACE_SIZE), dtype=np.float32)
    return manifest, ecg_array, face_array


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def build_features(output_dir=OUTPUT_DIR, max_samples=None):
    """Main entry point: build and save all features with time-matched labels.

    Args:
        output_dir: Directory for outputs.
        max_samples: If set, randomly subsample to at most this many samples.

    Returns:
        manifest: DataFrame of sample metadata + labels.
        ecg: (N, ECG_LENGTH) float32 array.
        face: (N, FACE_SIZE, FACE_SIZE) float32 array.
    """
    _ensure_dirs(output_dir)
    print("=" * 60)
    print("Exp2 Dataset Builder (v2: time-matched labels)")
    print("=" * 60)

    # Step 1: Build lab timeseries
    print("\n[1/3] Building lab measurement timeseries ...")
    lab_ts = _build_lab_timeseries(LAB_CSV)
    print(f"  → {len(lab_ts)} individual measurements")
    print(f"  → {lab_ts['hospital_id'].nunique()} unique hospital IDs")
    print(f"  → Analytes: {sorted(lab_ts['analyte'].unique())}")
    lab_ts.to_csv(os.path.join(output_dir, "lab_timeseries.csv"), index=False)

    # Step 2: Extract ECG + Face features with time-matched labels
    print("\n[2/3] Extracting ECG and face features with time-matched labels ...")
    manifest, ecg, face = _build_samples(DATA_ROOT, lab_ts, output_dir, max_samples)
    print(f"  → {len(manifest)} valid samples")
    print(f"  → {manifest['hospital_id'].nunique()} unique hospital IDs")

    # Step 3: Save
    print("\n[3/3] Saving features ...")
    targets_array = np.array(TARGETS, dtype=str)
    np.savez_compressed(
        os.path.join(output_dir, "features.npz"),
        sample_id=manifest["sample_id"].to_numpy(dtype=str),
        hospital_id=manifest["hospital_id"].to_numpy(dtype=str),
        ecg=ecg,
        face=face,
        targets=targets_array,
    )
    manifest.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)

    # Label summary
    summary_rows = []
    for t in TARGETS:
        vals = pd.to_numeric(manifest[t], errors="coerce").dropna()
        summary_rows.append({
            "target": t,
            "total": int(len(vals)),
            "positive": int(vals.sum()) if len(vals) > 0 else 0,
            "negative": int((1 - vals).sum()) if len(vals) > 0 else 0,
            "positive_rate": float(vals.mean()) if len(vals) > 0 else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "label_summary.csv"), index=False)

    print(f"\nDone. Outputs saved to {output_dir}/")
    return manifest, ecg, face


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp2 feature builder (v2: time-matched)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    build_features(output_dir=args.output_dir, max_samples=args.max_samples)
