"""Dataset builder for Exp2: time-matched ECG/face features and lab-test labels.

This version uses the regenerated merged_patient_info_*.csv files as the
patient-info source, performs unit-aware lab conversion, and keeps samples when
some labels are missing. Lab-derived labels are only assigned when the nearest
lab measurement is within LAB_MATCH_MAX_DELTA_HOURS of the ECG capture time.
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
    LAB_MATCH_MAX_DELTA_HOURS,
    OUTPUT_DIR,
    PATIENT_INFO_GLOB,
    PLACEHOLDER_HOSPITAL_IDS,
    SEED,
    TARGETS,
)

LAB_REPORT_TIMEZONE = "Asia/Shanghai"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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
    """Convert local Chinese hospital report times to UTC Unix seconds."""
    local_time = pd.to_datetime(series, errors="coerce")
    localized = local_time.dt.tz_localize(
        LAB_REPORT_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
    )
    unix = localized.astype("int64") // 10**9
    return pd.Series(unix, index=series.index).where(localized.notna(), np.nan)


def _first_nonempty(series):
    vals = series.astype(str).str.strip()
    vals = vals[(vals != "") & (vals != "nan") & (vals != "None")]
    return vals.iloc[0] if len(vals) else ""


def _is_valid_blood_pressure(low_bp, high_bp):
    """Return True for plausible diastolic/systolic BP values."""
    if pd.isna(low_bp) or pd.isna(high_bp):
        return False
    low_bp = float(low_bp)
    high_bp = float(high_bp)
    if low_bp <= 0 or high_bp <= 0:
        return False
    if not (30.0 <= low_bp <= 160.0 and 60.0 <= high_bp <= 260.0):
        return False
    if high_bp < low_bp:
        return False
    return True


def _mirror_from_patient_info_path(path):
    match = re.search(r"merged_patient_info_(\d+)\.csv$", os.path.basename(path))
    if not match:
        return None
    return f"mirror{int(match.group(1))}"


# ---------------------------------------------------------------------------
# Analyte definitions and thresholds
# ---------------------------------------------------------------------------

_ANALYTE_MAP = {
    "lactate": {
        "item_names": ["乳酸浓度"],
        "direction": "max",
    },
    "troponin": {
        "item_names": ["*肌钙蛋白Ⅰ(hsTnI)测定", "肌钙蛋白Ⅰ(hsTnI)测定"],
        "direction": "max",
    },
    "glucose": {
        "item_names": ["*葡萄糖(Glu)测定", "葡萄糖浓度"],
        "direction": "max",
    },
    "hemoglobin": {
        "item_names": ["*血红蛋白", "血红蛋白", "总血红蛋白"],
        "direction": "min",
    },
    "po2": {
        "item_names": ["氧分压", "患者体温下氧分压"],
        "direction": "min",
    },
    "pco2": {
        "item_names": ["二氧化碳分压", "患者体温下二氧化碳分压"],
        "direction": "max",
    },
}

_THRESHOLD_RULES = [
    ("lactate_high", "lactate", 2.0, "gt"),
    ("troponin_high", "troponin", 34.0, "gt"),
    ("glucose_high", "glucose", 7.8, "gt"),
    ("po2_low", "po2", 80.0, "lt"),
    ("pco2_low", "pco2", 34.0, "lt"),
    ("pco2_high", "pco2", 50.0, "gt"),
    ("lactate_moderate_high", "lactate", 4.0, "gt"),
    ("troponin_extreme_high", "troponin", 1000.0, "gt"),
    ("glucose_marked_high", "glucose", 10.0, "gt"),
    ("hemoglobin_moderate_low", "hemoglobin", 90.0, "lt"),
    ("po2_moderate_low", "po2", 70.0, "lt"),
]

_LAB_TARGET_TO_ANALYTE = {target: analyte for target, analyte, _, _ in _THRESHOLD_RULES}
_LAB_TARGET_TO_ANALYTE.update({
    "hemoglobin_low": "hemoglobin",
    "pco2_abnormal": "pco2",
})


def _apply_threshold(value, threshold, op):
    """Apply comparison operator."""
    if np.isnan(value):
        return np.nan
    if op == "gt":
        return int(value > threshold)
    if op == "lt":
        return int(value < threshold)
    return np.nan


def _standardize_lab_values(analyte_key, values, units):
    """Convert lab values to the threshold units used by this experiment."""
    values = pd.to_numeric(values, errors="coerce")
    unit = units.astype(str).str.lower().str.strip()

    if analyte_key == "glucose":
        # Thresholds are mmol/L. Convert mg/dL, keep mmol/L. Blank high values
        # are treated as likely mg/dL because physiologic mmol/L values are small.
        is_mgdl = unit.str.contains("mg/dl", regex=False) | ((unit == "") & (values > 40))
        return values.where(~is_mgdl, values / 18.0)

    if analyte_key == "hemoglobin":
        # Thresholds are g/L. Convert g/dL, keep g/L. Blank low values are
        # treated as likely g/dL because g/L values rarely fall below 25.
        is_gdl = unit.str.contains("g/dl", regex=False) | ((unit == "") & (values < 25))
        return values.where(~is_gdl, values * 10.0)

    return values


# ---------------------------------------------------------------------------
# Lab timeseries and metadata builders
# ---------------------------------------------------------------------------

def _read_lab_csv(lab_csv):
    df = pd.read_csv(lab_csv, dtype=str, keep_default_na=False)
    df["hospital_id"] = df["首页病案号"].apply(_normalize_hospital_id)
    df = df[df["hospital_id"] != ""].copy()
    df["timestamp_unix"] = _parse_datetime_to_unix(df["报告时间"])
    df = df.dropna(subset=["timestamp_unix"]).copy()
    return df


def _build_lab_timeseries_from_df(df):
    """Build a flat timeseries of individual lab measurements."""
    rows = []
    for analyte_key, info in _ANALYTE_MAP.items():
        subset = df[df["检验项名称"].isin(info["item_names"])].copy()
        if subset.empty:
            continue
        raw_value = _extract_numeric(subset["检验值(文本)"])
        subset["value"] = _standardize_lab_values(analyte_key, raw_value, subset["单位"])
        subset = subset.dropna(subset=["value"]).copy()
        for _, row in subset.iterrows():
            rows.append({
                "hospital_id": row["hospital_id"],
                "analyte": analyte_key,
                "value": float(row["value"]),
                "timestamp_unix": int(row["timestamp_unix"]),
                "unit": row.get("单位", ""),
                "item_name": row.get("检验项名称", ""),
            })
    return pd.DataFrame(rows)


def _build_lab_metadata_from_df(df):
    """Build patient-level metadata needed for labels/debug columns."""
    rows = []
    for hid, group in df.groupby("hospital_id"):
        rows.append({
            "hospital_id": hid,
            "sex": _first_nonempty(group["首页性别"]),
            "surgery_text": " ".join(group["首页手术操作名称"].dropna().astype(str).unique()),
            "admission_time": _first_nonempty(group["首页入院时间"]),
            "discharge_time": _first_nonempty(group["首页出院时间"]),
        })
    return pd.DataFrame(rows).set_index("hospital_id").to_dict(orient="index")


def _build_lab_timeseries(lab_csv):
    df = _read_lab_csv(lab_csv)
    return _build_lab_timeseries_from_df(df)


# ---------------------------------------------------------------------------
# Patient info readers
# ---------------------------------------------------------------------------

def _read_merged_patient_info(patient_info_glob=PATIENT_INFO_GLOB):
    """Build lookup: (mirror, lab_patient_id) -> patient info dict.

    Source files are the regenerated merged_patient_info_N.csv files. They have
    one row per raw mirror patient and include vitals/BP extracted from both
    patient_info.txt and the batch vitals extraction.
    """
    lookup = {}
    info_paths = sorted(glob.glob(patient_info_glob))
    if not info_paths:
        raise FileNotFoundError(f"No patient-info files matched {patient_info_glob}")

    for info_path in info_paths:
        mirror = _mirror_from_patient_info_path(info_path)
        if mirror is None:
            continue
        info = pd.read_csv(info_path, dtype=str, keep_default_na=False)
        required = {"lab_patient_id", "hospital_patient_id"}
        missing = required - set(info.columns)
        if missing:
            raise ValueError(f"{info_path} missing required columns: {sorted(missing)}")
        for _, row in info.iterrows():
            lab_patient_id = pd.to_numeric(row["lab_patient_id"], errors="coerce")
            if pd.isna(lab_patient_id):
                continue
            key = (mirror, int(lab_patient_id))
            lookup[key] = {
                "Lab_Patient_ID": str(int(lab_patient_id)),
                "Hospital_Patient_ID": row.get("hospital_patient_id", ""),
                "Low_Blood_Pressure": row.get("low_blood_pressure", -1),
                "High_Blood_Pressure": row.get("high_blood_pressure", -1),
                "blood_oxygen": row.get("blood_oxygen", -1),
                "heart_rate": row.get("heart_rate", -1),
                "respiratory_rate": row.get("respiratory_rate", -1),
                "temperature": row.get("temperature", -1),
                "patient_info_source": os.path.basename(info_path),
            }
    return lookup


# Backward-compatible alias for existing imports/tests.
def _read_cleaned_info(data_root):
    return _read_merged_patient_info()


# ---------------------------------------------------------------------------
# Signal / frame readers
# ---------------------------------------------------------------------------

def _parse_signal_file(path):
    """Parse patient ID and session ID from signal filename."""
    name = os.path.basename(path)
    match = re.match(r"patient_(\d+)_(\d+)\.csv$", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _get_session_timestamp(signal_path):
    """Get the capture timestamp (median) of an ECG session, as Unix seconds."""
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
    """Load/cache face frame, return normalized grayscale (32x32)."""
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

    std = float(np.std(vector))
    if std <= 1e-8:
        vector = vector - float(np.mean(vector))
    else:
        vector = (vector - float(np.mean(vector))) / std
    return vector.astype(np.float32)


# ---------------------------------------------------------------------------
# Time-matched label computation
# ---------------------------------------------------------------------------

def _find_closest_measurement(lab_ts, session_time, analyte_key,
                              max_delta_hours=LAB_MATCH_MAX_DELTA_HOURS):
    """Find closest lab measurement within a maximum absolute time delta.

    Returns:
        (value, abs_delta_hours, signed_delta_hours, lab_timestamp_unix)
        or (nan, nan, nan, nan) if no in-window measurement exists.
    """
    subset = lab_ts[lab_ts["analyte"] == analyte_key]
    if subset.empty:
        return np.nan, np.nan, np.nan, np.nan

    lab_times = subset["timestamp_unix"].to_numpy(np.float64)
    signed_sec = lab_times - session_time
    abs_sec = np.abs(signed_sec)
    best_idx = int(np.argmin(abs_sec))
    abs_delta_h = float(abs_sec[best_idx]) / 3600.0
    signed_delta_h = float(signed_sec[best_idx]) / 3600.0
    if max_delta_hours is not None and abs_delta_h > max_delta_hours:
        return np.nan, abs_delta_h, signed_delta_h, float(lab_times[best_idx])
    value = float(subset.iloc[best_idx]["value"])
    return value, abs_delta_h, signed_delta_h, float(lab_times[best_idx])


def _compute_labels_for_session(lab_ts, session_time, sex_value,
                                max_delta_hours=LAB_MATCH_MAX_DELTA_HOURS):
    """Compute all binary lab labels for one ECG capture session."""
    labels = {target: np.nan for target in TARGETS}
    raw_values = {}
    time_deltas = {}
    signed_time_deltas = {}
    lab_timestamps = {}

    analyte_values = {}
    for analyte_key in _ANALYTE_MAP:
        val, delta_h, signed_delta_h, lab_time = _find_closest_measurement(
            lab_ts, session_time, analyte_key, max_delta_hours=max_delta_hours)
        analyte_values[analyte_key] = val
        raw_values[analyte_key] = val
        time_deltas[analyte_key] = delta_h
        signed_time_deltas[analyte_key] = signed_delta_h
        lab_timestamps[analyte_key] = lab_time

    for target_name, analyte_key, threshold, op in _THRESHOLD_RULES:
        labels[target_name] = _apply_threshold(analyte_values.get(analyte_key, np.nan), threshold, op)

    hb_val = analyte_values.get("hemoglobin", np.nan)
    hb_threshold = 130.0 if sex_value == "男" else 120.0
    labels["hemoglobin_low"] = _apply_threshold(hb_val, hb_threshold, "lt")

    pco2_val = analyte_values.get("pco2", np.nan)
    if not np.isnan(pco2_val):
        labels["pco2_abnormal"] = int((pco2_val < 35.0) or (pco2_val > 45.0))

    # Current available inputs do not provide a meaningful non-degenerate
    # coronary-context target. Leave it missing so training skips it.
    labels["coronary_context"] = np.nan

    return labels, raw_values, time_deltas, signed_time_deltas, lab_timestamps


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def _build_samples(data_root, lab_timeseries, lab_metadata, output_dir, max_samples=None):
    """Iterate over signal files, extract ECG + face, compute labels."""
    info_lookup = _read_merged_patient_info()

    lab_by_hospital = {hid: group for hid, group in lab_timeseries.groupby("hospital_id")}

    manifest_rows = []
    ecg_list = []
    face_list = []
    failures = []
    skip_counts = {
        "parse_fail": 0,
        "missing_patient_info": 0,
        "placeholder_hospital_id": 0,
        "missing_timestamp": 0,
        "feature_load_failed": 0,
    }

    signal_paths = sorted(
        glob.glob(os.path.join(data_root, "mirror*_auto_cleaned_sqi", "patient_*.csv"))
    )
    rng = np.random.default_rng(SEED)
    if max_samples is not None and len(signal_paths) > max_samples:
        signal_paths = list(rng.choice(signal_paths, size=max_samples, replace=False))
        signal_paths.sort()

    for signal_path in signal_paths:
        parsed = _parse_signal_file(signal_path)
        if parsed is None:
            skip_counts["parse_fail"] += 1
            continue
        lab_patient_id, session_id = parsed
        mirror = os.path.basename(os.path.dirname(signal_path)).split("_")[0]

        info = info_lookup.get((mirror, lab_patient_id))
        if info is None:
            skip_counts["missing_patient_info"] += 1
            continue

        hospital_id = _normalize_hospital_id(info.get("Hospital_Patient_ID", ""))
        if hospital_id == "":
            skip_counts["placeholder_hospital_id"] += 1
            continue

        session_time = _get_session_timestamp(signal_path)
        if session_time is None:
            skip_counts["missing_timestamp"] += 1
            continue

        sample_id = f"{mirror}_patient_{lab_patient_id:06d}_{session_id}"
        video_path = os.path.join(
            data_root, f"{mirror}_data", f"patient_{lab_patient_id:06d}", "video.avi"
        )

        try:
            ecg_vec = _load_ecg(signal_path, ECG_LENGTH, ECG_WINDOW_SEC)
            face_mat = _load_face(video_path, sample_id, output_dir, FACE_SIZE)
        except Exception as exc:
            skip_counts["feature_load_failed"] += 1
            failures.append({"sample_id": sample_id, "error": str(exc)})
            continue

        metadata = lab_metadata.get(hospital_id, {})
        sex_value = metadata.get("sex", "")
        patient_lab = lab_by_hospital.get(hospital_id, pd.DataFrame(columns=lab_timeseries.columns))
        labels, raw_vals, time_deltas, signed_deltas, lab_times = _compute_labels_for_session(
            patient_lab, session_time, sex_value)

        low_bp = pd.to_numeric(info.get("Low_Blood_Pressure", -1), errors="coerce")
        high_bp = pd.to_numeric(info.get("High_Blood_Pressure", -1), errors="coerce")
        bp_is_valid = _is_valid_blood_pressure(low_bp, high_bp)
        if bp_is_valid:
            labels["high_blood_pressure"] = int(high_bp >= 140.0 or low_bp >= 90.0)

        row_data = {
            "sample_id": sample_id,
            "hospital_id": hospital_id,
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "session_id": session_id,
            "capture_time_unix": session_time,
            "sex": sex_value,
            "low_blood_pressure": float(low_bp) if pd.notna(low_bp) else np.nan,
            "high_blood_pressure": float(high_bp) if pd.notna(high_bp) else np.nan,
            "blood_pressure_valid": int(bp_is_valid),
            "patient_info_source": info.get("patient_info_source", ""),
            "admission_time": metadata.get("admission_time", ""),
            "discharge_time": metadata.get("discharge_time", ""),
        }
        row_data.update(labels)
        for analyte_key in _ANALYTE_MAP:
            row_data[f"{analyte_key}_value"] = raw_vals.get(analyte_key, np.nan)
            row_data[f"{analyte_key}_delta_h"] = time_deltas.get(analyte_key, np.nan)
            row_data[f"{analyte_key}_signed_delta_h"] = signed_deltas.get(analyte_key, np.nan)
            row_data[f"{analyte_key}_lab_time_unix"] = lab_times.get(analyte_key, np.nan)
        manifest_rows.append(row_data)
        ecg_list.append(ecg_vec)
        face_list.append(face_mat)

    print("  Skip counts:")
    for key, value in skip_counts.items():
        print(f"    {key}: {value}")
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(output_dir=OUTPUT_DIR, max_samples=None):
    """Main entry point: build and save all features with time-matched labels."""
    _ensure_dirs(output_dir)
    print("=" * 60)
    print("Exp2 Dataset Builder (merged patient info + bounded time-matched labels)")
    print("=" * 60)

    print("\n[1/3] Building lab measurement timeseries ...")
    lab_df = _read_lab_csv(LAB_CSV)
    lab_ts = _build_lab_timeseries_from_df(lab_df)
    lab_metadata = _build_lab_metadata_from_df(lab_df)
    print(f"  -> {len(lab_ts)} individual measurements")
    print(f"  -> {lab_ts['hospital_id'].nunique()} unique hospital IDs")
    print(f"  -> Analytes: {sorted(lab_ts['analyte'].unique())}")
    print(f"  -> Lab label max delta: {LAB_MATCH_MAX_DELTA_HOURS:g} hours")
    lab_ts.to_csv(os.path.join(output_dir, "lab_timeseries.csv"), index=False)

    print("\n[2/3] Extracting ECG and face features with bounded time-matched labels ...")
    manifest, ecg, face = _build_samples(DATA_ROOT, lab_ts, lab_metadata, output_dir, max_samples)
    print(f"  -> {len(manifest)} valid samples")
    print(f"  -> {manifest['hospital_id'].nunique()} unique hospital IDs")

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
    parser = argparse.ArgumentParser(description="Exp2 feature builder")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    build_features(output_dir=args.output_dir, max_samples=args.max_samples)
