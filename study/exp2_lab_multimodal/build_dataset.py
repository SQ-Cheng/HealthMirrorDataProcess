"""Dataset builder for Exp2: extracts ECG/face features and lab-test labels.

Produces:
    outputs/manifest.csv   — sample-level metadata and binary labels
    outputs/features.npz   — {'sample_id', 'ecg', 'face', 'targets'}
    outputs/label_summary.csv — per-target statistics
"""

import argparse
import glob
import json
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


def _values_for_items(group, item_names, unit_converter=None):
    """Extract numeric values for given test item names from a patient group."""
    rows = group[group["检验项名称"].isin(item_names)].copy()
    if rows.empty:
        return np.array([], dtype=np.float64)
    vals = _extract_numeric(rows["检验值(文本)"])
    if unit_converter is not None:
        vals = unit_converter(vals)
    return vals.dropna().to_numpy(dtype=np.float64)


def _glucose_to_mmol(values):
    """Convert glucose mg/dL → mmol/L (divide by 18.0)."""
    return values / 18.0


def _hemoglobin_to_gl(values):
    """Convert hemoglobin g/dL → g/L (multiply by 10)."""
    return values * 10.0


# ═══════════════════════════════════════════════════════════════════════
# Lab label builder
# ═══════════════════════════════════════════════════════════════════════

def _build_lab_labels(lab_csv):
    """Build per-hospital_id lab test abnormality labels.

    Returns DataFrame with hospital_id + TARGET columns.
    """
    df = pd.read_csv(lab_csv, dtype=str, keep_default_na=False)
    df["hospital_id"] = df["首页病案号"].apply(_normalize_hospital_id)
    df = df[df["hospital_id"] != ""].copy()

    rows = []
    for hospital_id, group in df.groupby("hospital_id"):
        sex_value = group["首页性别"].iloc[0] if len(group) > 0 else ""
        text_blob = " ".join(group["首页手术操作名称"].dropna().astype(str))

        lactate = _values_for_items(group, ["乳酸浓度"])
        troponin = _values_for_items(group, [
            "*肌钙蛋白Ⅰ(hsTnI)测定", "肌钙蛋白Ⅰ(hsTnI)测定"
        ])
        glucose = _values_for_items(group, ["*葡萄糖(Glu)测定"])
        if glucose.size == 0:
            glucose = _values_for_items(group, ["葡萄糖浓度"], _glucose_to_mmol)
        hemoglobin = _values_for_items(
            group, ["*血红蛋白", "血红蛋白", "总血红蛋白"], _hemoglobin_to_gl
        )
        po2 = _values_for_items(group, ["氧分压", "患者体温下氧分压"])
        pco2 = _values_for_items(group, ["二氧化碳分压", "患者体温下二氧化碳分压"])

        hb_threshold = 130.0 if sex_value == "男" else 120.0

        def _has(vals):
            return vals.size > 0

        rows.append({
            "hospital_id": hospital_id,
            "sex": sex_value,
            "lab_rows": int(len(group)),
            "lactate_max": float(np.max(lactate)) if _has(lactate) else np.nan,
            "troponin_max": float(np.max(troponin)) if _has(troponin) else np.nan,
            "glucose_max_mmol": float(np.max(glucose)) if _has(glucose) else np.nan,
            "hemoglobin_min_gl": float(np.min(hemoglobin)) if _has(hemoglobin) else np.nan,
            "po2_min_mmhg": float(np.min(po2)) if _has(po2) else np.nan,
            "pco2_min_mmhg": float(np.min(pco2)) if _has(pco2) else np.nan,
            "pco2_max_mmhg": float(np.max(pco2)) if _has(pco2) else np.nan,
            "lactate_high": int(np.max(lactate) > 2.0) if _has(lactate) else np.nan,
            "troponin_high": int(np.max(troponin) > 34.0) if _has(troponin) else np.nan,
            "glucose_high": int(np.max(glucose) > 7.8) if _has(glucose) else np.nan,
            "hemoglobin_low": int(np.min(hemoglobin) < hb_threshold) if _has(hemoglobin) else np.nan,
            "po2_low": int(np.min(po2) < 80.0) if _has(po2) else np.nan,
            "pco2_abnormal": int((np.min(pco2) < 35.0) or (np.max(pco2) > 45.0))
            if _has(pco2) else np.nan,
            "coronary_context": int("冠心病" in text_blob),
            "lactate_moderate_high": int(np.max(lactate) > 4.0) if _has(lactate) else np.nan,
            "troponin_extreme_high": int(np.max(troponin) > 1000.0) if _has(troponin) else np.nan,
            "glucose_marked_high": int(np.max(glucose) > 10.0) if _has(glucose) else np.nan,
            "hemoglobin_moderate_low": int(np.min(hemoglobin) < 90.0) if _has(hemoglobin) else np.nan,
            "po2_moderate_low": int(np.min(po2) < 70.0) if _has(po2) else np.nan,
            "pco2_low": int(np.min(pco2) < 34.0) if _has(pco2) else np.nan,
            "pco2_high": int(np.max(pco2) > 50.0) if _has(pco2) else np.nan,
            # high_blood_pressure is computed per-sample from cleaned_patient_info
            # (BP values are session-level, not hospital-level)
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
# Main dataset builder
# ═══════════════════════════════════════════════════════════════════════

def _build_samples(data_root, lab_labels, output_dir, max_samples=None):
    """Iterate over all signal files, extract ECG + face, match labels.

    Returns:
        manifest: DataFrame with metadata and labels.
        ecg_array:  (N, ECG_LENGTH) float32
        face_array: (N, FACE_SIZE, FACE_SIZE) float32
    """
    info_lookup = _read_cleaned_info(data_root)
    lab_lookup = lab_labels.set_index("hospital_id")

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
        if hospital_id == "" or hospital_id not in lab_lookup.index:
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

        lab_row = lab_lookup.loc[hospital_id]

        # Compute high_blood_pressure from patient info (session-level BP)
        low_bp = pd.to_numeric(info.get("Low_Blood_Pressure", -1), errors="coerce")
        high_bp = pd.to_numeric(info.get("High_Blood_Pressure", -1), errors="coerce")
        if pd.notna(high_bp) and pd.notna(low_bp) and high_bp > 0 and low_bp > 0:
            hbp_label = int(high_bp >= 140.0 or low_bp >= 90.0)
        else:
            hbp_label = np.nan

        row_data = {
            "sample_id": sample_id,
            "hospital_id": hospital_id,
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "session_id": session_id,
        }
        for t in TARGETS:
            if t == "high_blood_pressure":
                row_data[t] = hbp_label
            else:
                row_data[t] = lab_row[t]
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
    """Main entry point: build and save all features.

    Args:
        output_dir: Directory for outputs (manifest.csv, features.npz, etc.).
        max_samples: If set, randomly subsample to at most this many samples.

    Returns:
        manifest: DataFrame of sample metadata + labels.
        ecg: (N, ECG_LENGTH) float32 array.
        face: (N, FACE_SIZE, FACE_SIZE) float32 array.
    """
    _ensure_dirs(output_dir)
    print("=" * 60)
    print("Exp2 Dataset Builder")
    print("=" * 60)

    # Step 1: Build lab labels
    print("\n[1/3] Building lab test labels ...")
    lab_labels = _build_lab_labels(LAB_CSV)
    print(f"  → {len(lab_labels)} unique hospital IDs with lab data")
    lab_labels.to_csv(os.path.join(output_dir, "lab_labels.csv"), index=False)

    # Step 2: Extract ECG + Face features
    print("\n[2/3] Extracting ECG and face features ...")
    manifest, ecg, face = _build_samples(DATA_ROOT, lab_labels, output_dir, max_samples)
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
            "positive": int(vals.sum()),
            "negative": int((1 - vals).sum()),
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
