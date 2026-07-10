"""Build face-only Exp2 features with corrected label matching."""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from study.exp2_lab_multimodal.build_dataset import (
    _ANALYTE_MAP,
    _build_lab_metadata_from_df,
    _build_lab_timeseries_from_df,
    _compute_labels_for_session,
    _get_session_timestamp,
    _is_valid_blood_pressure,
    _load_face,
    _normalize_hospital_id,
    _parse_signal_file,
    _read_lab_csv,
    _read_merged_patient_info,
)

from .config import DATA_ROOT, FACE_SIZE, LAB_CSV, OUTPUT_DIR, SEED, TARGETS


def _ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def build_features(output_dir=OUTPUT_DIR, max_samples=None):
    """Build face-only feature arrays and labels.

    ECG signal files are used only to identify valid cleaned sessions and their
    capture timestamps. The saved feature file contains a single face image per
    sample and no ECG tensor.
    """
    _ensure_dirs(output_dir)
    print("=" * 60)
    print("Exp2 Face-Only Dataset Builder")
    print("=" * 60)

    print("\n[1/3] Building lab timeseries and metadata ...")
    lab_df = _read_lab_csv(LAB_CSV)
    lab_ts = _build_lab_timeseries_from_df(lab_df)
    lab_metadata = _build_lab_metadata_from_df(lab_df)
    lab_ts.to_csv(os.path.join(output_dir, "lab_timeseries.csv"), index=False)
    lab_by_hospital = {hid: group for hid, group in lab_ts.groupby("hospital_id")}
    print(f"  -> {len(lab_ts)} lab measurements, {lab_ts['hospital_id'].nunique()} hospital IDs")

    print("\n[2/3] Extracting one face frame per cleaned session ...")
    info_lookup = _read_merged_patient_info()
    signal_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "mirror*_auto_cleaned_sqi", "patient_*.csv")))
    rng = np.random.default_rng(SEED)
    if max_samples is not None and len(signal_paths) > max_samples:
        signal_paths = sorted(rng.choice(signal_paths, size=max_samples, replace=False).tolist())

    manifest_rows, face_list = [], []
    skip_counts = {
        "parse_fail": 0,
        "missing_patient_info": 0,
        "placeholder_hospital_id": 0,
        "missing_timestamp": 0,
        "face_load_failed": 0,
    }

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
        video_path = os.path.join(DATA_ROOT, f"{mirror}_data", f"patient_{lab_patient_id:06d}", "video.avi")
        try:
            face_mat = _load_face(video_path, sample_id, output_dir, FACE_SIZE)
        except Exception:
            skip_counts["face_load_failed"] += 1
            continue

        metadata = lab_metadata.get(hospital_id, {})
        patient_lab = lab_by_hospital.get(hospital_id, pd.DataFrame(columns=lab_ts.columns))
        labels, raw_vals, time_deltas, signed_deltas, lab_times = _compute_labels_for_session(
            patient_lab, session_time, metadata.get("sex", ""))

        low_bp = pd.to_numeric(info.get("Low_Blood_Pressure", -1), errors="coerce")
        high_bp = pd.to_numeric(info.get("High_Blood_Pressure", -1), errors="coerce")
        bp_is_valid = _is_valid_blood_pressure(low_bp, high_bp)
        if bp_is_valid:
            labels["high_blood_pressure"] = int(high_bp >= 140.0 or low_bp >= 90.0)

        row = {
            "sample_id": sample_id,
            "hospital_id": hospital_id,
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "session_id": session_id,
            "capture_time_unix": session_time,
            "sex": metadata.get("sex", ""),
            "low_blood_pressure": float(low_bp) if pd.notna(low_bp) else np.nan,
            "high_blood_pressure": float(high_bp) if pd.notna(high_bp) else np.nan,
            "blood_pressure_valid": int(bp_is_valid),
            "patient_info_source": info.get("patient_info_source", ""),
        }
        row.update(labels)
        for analyte in _ANALYTE_MAP:
            row[f"{analyte}_value"] = raw_vals.get(analyte, np.nan)
            row[f"{analyte}_delta_h"] = time_deltas.get(analyte, np.nan)
            row[f"{analyte}_signed_delta_h"] = signed_deltas.get(analyte, np.nan)
            row[f"{analyte}_lab_time_unix"] = lab_times.get(analyte, np.nan)
        manifest_rows.append(row)
        face_list.append(face_mat)

    print("  Skip counts:")
    for key, value in skip_counts.items():
        print(f"    {key}: {value}")

    manifest = pd.DataFrame(manifest_rows)
    face = np.stack(face_list, axis=0) if face_list else np.empty((0, FACE_SIZE, FACE_SIZE), dtype=np.float32)
    print(f"  -> {len(manifest)} samples, {manifest['hospital_id'].nunique() if len(manifest) else 0} patients")

    print("\n[3/3] Saving face-only features ...")
    np.savez_compressed(
        os.path.join(output_dir, "features.npz"),
        sample_id=manifest["sample_id"].to_numpy(dtype=str),
        hospital_id=manifest["hospital_id"].to_numpy(dtype=str),
        face=face,
        targets=np.array(TARGETS, dtype=str),
    )
    manifest.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)

    summary_rows = []
    for target in TARGETS:
        vals = pd.to_numeric(manifest[target], errors="coerce").dropna()
        summary_rows.append({
            "target": target,
            "total": int(len(vals)),
            "positive": int(vals.sum()) if len(vals) else 0,
            "negative": int((1 - vals).sum()) if len(vals) else 0,
            "positive_rate": float(vals.mean()) if len(vals) else np.nan,
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "label_summary.csv"), index=False)
    print(f"Done. Outputs saved to {output_dir}/")
    return manifest, face


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Exp2 face-only features")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    build_features(output_dir=args.output_dir, max_samples=args.max_samples)
