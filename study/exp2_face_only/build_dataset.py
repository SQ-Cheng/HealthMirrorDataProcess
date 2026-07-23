"""Build 20 native-resolution non-adjacent RGB samples per matched event."""

import argparse
import glob
import json
import os
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from study.exp2_lab_multimodal.build_dataset import (
    LAB_REPORT_TIMEZONE,
    _ANALYTE_MAP,
    _build_lab_metadata_from_df,
    _build_lab_timeseries_from_df,
    _compute_labels_for_session,
    _get_session_timestamp,
    _is_valid_blood_pressure,
    _normalize_hospital_id,
    _parse_signal_file,
    _read_lab_csv,
    _read_merged_patient_info,
)

from .config import (
    DATA_ROOT,
    FACE_SIZE,
    FRAME_QUANTILES,
    LAB_CSV,
    LAB_MATCH_MAX_DELTA_HOURS,
    MIN_SOURCE_FRAME_GAP,
    NUM_FACE_FRAMES,
    OUTPUT_DIR,
    SEED,
    TARGETS,
)


def _ensure_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb_frames_aug20"), exist_ok=True)


def _build_video_table(info_lookup):
    """Collapse cleaned ECG sessions to one physical video time point."""
    signal_paths = sorted(
        glob.glob(os.path.join(DATA_ROOT, "mirror*_auto_cleaned_sqi", "patient_*.csv"))
    )
    grouped_times = {}
    skip_counts = {
        "signal_parse_fail": 0,
        "missing_patient_info": 0,
        "placeholder_hospital_id": 0,
        "missing_timestamp": 0,
        "missing_video": 0,
    }

    for signal_path in signal_paths:
        parsed = _parse_signal_file(signal_path)
        if parsed is None:
            skip_counts["signal_parse_fail"] += 1
            continue
        lab_patient_id, _ = parsed
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
        grouped_times.setdefault((mirror, lab_patient_id, hospital_id), []).append(session_time)

    rows = []
    for (mirror, lab_patient_id, hospital_id), times in sorted(grouped_times.items()):
        video_path = os.path.join(
            DATA_ROOT, f"{mirror}_data", f"patient_{lab_patient_id:06d}", "video.avi"
        )
        if not os.path.exists(video_path):
            skip_counts["missing_video"] += 1
            continue
        info = info_lookup[(mirror, lab_patient_id)]
        low_bp = pd.to_numeric(info.get("Low_Blood_Pressure", np.nan), errors="coerce")
        high_bp = pd.to_numeric(info.get("High_Blood_Pressure", np.nan), errors="coerce")
        rows.append({
            "video_id": f"{mirror}_patient_{lab_patient_id:06d}",
            "hospital_id": hospital_id,
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "capture_time_unix": float(np.median(times)),
            "capture_start_unix": float(np.min(times)),
            "capture_end_unix": float(np.max(times)),
            "clean_session_count": len(times),
            "video_path": video_path,
            "low_blood_pressure": float(low_bp) if pd.notna(low_bp) else np.nan,
            "high_blood_pressure": float(high_bp) if pd.notna(high_bp) else np.nan,
            "blood_pressure_valid": int(_is_valid_blood_pressure(low_bp, high_bp)),
            "patient_info_source": info.get("patient_info_source", ""),
        })
    return pd.DataFrame(rows), skip_counts


def _jpeg_ranges(video_path):
    with open(video_path, "rb") as handle:
        data = handle.read()
    ranges = []
    position = 0
    while True:
        start = data.find(b"\xff\xd8", position)
        if start < 0:
            break
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        ranges.append((start, end + 2))
        position = end + 2
    return data, ranges


def _decode_nearest_frame(data, ranges, target_index):
    """Decode a desired MJPEG frame, searching nearby if that frame is corrupt."""
    offsets = [0]
    for distance in range(1, 31):
        offsets.extend((-distance, distance))
    for offset in offsets:
        index = target_index + offset
        if index < 0 or index >= len(ranges):
            continue
        start, end = ranges[index]
        try:
            image = Image.open(BytesIO(data[start:end])).convert("RGB")
            image.load()
            return image, index
        except Exception:
            continue
    raise ValueError(f"No decodable JPEG near frame {target_index}")


def _load_rgb_frame_group(video_path, video_id, output_dir):
    """Return 20 native-resolution RGB frames as uint8 [T, C, H, W]."""
    frame_dir = os.path.join(output_dir, "rgb_frames_aug20")
    cache_paths = [
        os.path.join(frame_dir, f"{video_id}_frame_{index + 1:02d}.png")
        for index in range(NUM_FACE_FRAMES)
    ]
    source_indices_path = os.path.join(frame_dir, f"{video_id}_source_indices.csv")
    if all(os.path.exists(path) for path in cache_paths):
        images = [Image.open(path).convert("RGB") for path in cache_paths]
    else:
        data, ranges = _jpeg_ranges(video_path)
        if not ranges:
            raise ValueError("No JPEG frames found in MJPEG video")
        target_indices = [
            int(round(quantile * (len(ranges) - 1))) for quantile in FRAME_QUANTILES
        ]
        if len(target_indices) != NUM_FACE_FRAMES:
            raise ValueError("FRAME_QUANTILES and NUM_FACE_FRAMES disagree")
        if min(np.diff(target_indices)) < MIN_SOURCE_FRAME_GAP:
            target_indices = np.round(
                np.linspace(0, len(ranges) - 1, NUM_FACE_FRAMES)
            ).astype(int).tolist()
        if min(np.diff(target_indices)) < MIN_SOURCE_FRAME_GAP:
            raise ValueError("video is too short for 20 non-adjacent frames")
        images = []
        used_indices = []
        for target_index, cache_path in zip(target_indices, cache_paths):
            image, decoded_index = _decode_nearest_frame(data, ranges, target_index)
            if image.size != (FACE_SIZE, FACE_SIZE):
                raise ValueError(
                    f"native frame size {image.size} differs from expected "
                    f"{(FACE_SIZE, FACE_SIZE)}"
                )
            image.save(cache_path, format="PNG", compress_level=1)
            images.append(image)
            used_indices.append(decoded_index)
        selected_indices = np.asarray(sorted(used_indices), dtype=np.int64)
        if len(np.unique(selected_indices)) != NUM_FACE_FRAMES:
            raise ValueError("decoded source frames are not unique")
        if min(np.diff(selected_indices)) < MIN_SOURCE_FRAME_GAP:
            raise ValueError("decoded source frames are adjacent or too close")
        np.savetxt(source_indices_path, selected_indices, fmt="%d")
    invalid_sizes = sorted({
        image.size for image in images if image.size != (FACE_SIZE, FACE_SIZE)
    })
    if invalid_sizes:
        raise ValueError(
            f"cached frame sizes are not native {FACE_SIZE}x{FACE_SIZE}: {invalid_sizes}"
        )
    arrays = [np.asarray(image, dtype=np.uint8) for image in images]
    return np.stack(arrays, axis=0).transpose(0, 3, 1, 2)


def _empty_label_dict():
    return {target: np.nan for target in TARGETS}


def _expand_event_frames(base_manifest):
    """Create exactly one data row per selected frame for every base event."""
    if base_manifest.empty:
        expanded = base_manifest.copy()
        expanded["base_event_id"] = pd.Series(dtype=str)
        expanded["frame_index"] = pd.Series(dtype=np.int64)
        return expanded
    expanded = base_manifest.loc[base_manifest.index.repeat(NUM_FACE_FRAMES)].copy()
    expanded["base_event_id"] = expanded["sample_id"].astype(str)
    expanded["frame_index"] = np.tile(np.arange(NUM_FACE_FRAMES), len(base_manifest))
    expanded["sample_id"] = (
        expanded["base_event_id"]
        + "_frame_"
        + (expanded["frame_index"] + 1).astype(str).str.zfill(2)
    )
    return expanded.reset_index(drop=True)


def _build_event_manifest(video_table, lab_timeseries, lab_metadata):
    """Match each lab event to its nearest video, then add one BP event per video."""
    videos_by_hospital = {
        hospital_id: group.reset_index(drop=True)
        for hospital_id, group in video_table.groupby("hospital_id")
    }
    rows = []
    skip_counts = {
        "lab_event_without_video": 0,
        "lab_event_outside_time_window": 0,
    }

    event_groups = lab_timeseries.groupby(["hospital_id", "timestamp_unix"], sort=True)
    for (hospital_id, event_time), event_labs in event_groups:
        candidates = videos_by_hospital.get(hospital_id)
        if candidates is None or candidates.empty:
            skip_counts["lab_event_without_video"] += 1
            continue
        candidate_times = candidates["capture_time_unix"].to_numpy(dtype=np.float64)
        nearest_index = int(np.argmin(np.abs(candidate_times - float(event_time))))
        video = candidates.iloc[nearest_index]
        signed_delta_h = (float(event_time) - float(video["capture_time_unix"])) / 3600.0
        delta_h = abs(signed_delta_h)
        if delta_h > LAB_MATCH_MAX_DELTA_HOURS:
            skip_counts["lab_event_outside_time_window"] += 1
            continue

        metadata = lab_metadata.get(hospital_id, {})
        labels, raw_values, _, _, lab_times = _compute_labels_for_session(
            event_labs,
            float(event_time),
            metadata.get("sex", ""),
            max_delta_hours=0.0,
        )
        # BP is a video-time observation and is represented once in a separate row.
        labels["high_blood_pressure"] = np.nan
        row = {
            "sample_id": f"lab_{hospital_id}_{int(float(event_time))}",
            "event_type": "lab",
            "hospital_id": hospital_id,
            "video_id": video["video_id"],
            "mirror": video["mirror"],
            "lab_patient_id": int(video["lab_patient_id"]),
            "capture_time_unix": float(video["capture_time_unix"]),
            "label_time_unix": float(event_time),
            "match_delta_h": delta_h,
            "match_signed_delta_h": signed_delta_h,
            "sex": metadata.get("sex", ""),
            "low_blood_pressure": np.nan,
            "high_blood_pressure": np.nan,
            "blood_pressure_valid": 0,
            "patient_info_source": video["patient_info_source"],
        }
        row.update(labels)
        for analyte in _ANALYTE_MAP:
            value = raw_values.get(analyte, np.nan)
            row[f"{analyte}_value"] = value
            if pd.notna(value):
                row[f"{analyte}_delta_h"] = delta_h
                row[f"{analyte}_signed_delta_h"] = signed_delta_h
                row[f"{analyte}_lab_time_unix"] = lab_times.get(analyte, event_time)
            else:
                row[f"{analyte}_delta_h"] = np.nan
                row[f"{analyte}_signed_delta_h"] = np.nan
                row[f"{analyte}_lab_time_unix"] = np.nan
        rows.append(row)

    for video in video_table.itertuples(index=False):
        if not bool(video.blood_pressure_valid):
            continue
        labels = _empty_label_dict()
        labels["high_blood_pressure"] = int(
            float(video.high_blood_pressure) >= 140.0
            or float(video.low_blood_pressure) >= 90.0
        )
        row = {
            "sample_id": f"bp_{video.video_id}",
            "event_type": "blood_pressure",
            "hospital_id": video.hospital_id,
            "video_id": video.video_id,
            "mirror": video.mirror,
            "lab_patient_id": int(video.lab_patient_id),
            "capture_time_unix": float(video.capture_time_unix),
            "label_time_unix": float(video.capture_time_unix),
            "match_delta_h": 0.0,
            "match_signed_delta_h": 0.0,
            "sex": lab_metadata.get(video.hospital_id, {}).get("sex", ""),
            "low_blood_pressure": float(video.low_blood_pressure),
            "high_blood_pressure": float(video.high_blood_pressure),
            "blood_pressure_valid": 1,
            "patient_info_source": video.patient_info_source,
        }
        row.update(labels)
        for analyte in _ANALYTE_MAP:
            row[f"{analyte}_value"] = np.nan
            row[f"{analyte}_delta_h"] = np.nan
            row[f"{analyte}_signed_delta_h"] = np.nan
            row[f"{analyte}_lab_time_unix"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows), skip_counts


def _label_summary(manifest):
    rows = []
    for target in TARGETS:
        values = pd.to_numeric(manifest[target], errors="coerce").dropna()
        rows.append({
            "target": target,
            "total": int(len(values)),
            "patients": int(manifest.loc[values.index, "hospital_id"].nunique()),
            "positive": int(values.sum()) if len(values) else 0,
            "negative": int((1 - values).sum()) if len(values) else 0,
            "positive_rate": float(values.mean()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def _mask_conflicting_hemoglobin_video_labels(base_manifest):
    """Mask a target when one video has both binary labels for that target."""
    result = base_manifest.copy()
    audit_rows = []
    targets = ("hemoglobin_low", "hemoglobin_moderate_low")
    for target in targets:
        labels = pd.to_numeric(result[target], errors="coerce")
        valid = result.loc[labels.notna(), ["video_id", "hospital_id"]].copy()
        valid["label"] = labels.loc[valid.index].astype(int)
        for video_id, group in valid.groupby("video_id", sort=True):
            positive_count = int(group["label"].eq(1).sum())
            negative_count = int(group["label"].eq(0).sum())
            if positive_count == 0 or negative_count == 0:
                continue
            mask = result["video_id"].eq(video_id) & result[target].notna()
            masked_event_count = int(mask.sum())
            result.loc[mask, target] = np.nan
            audit_rows.append({
                "issue_type": "conflicting_labels_for_identical_video",
                "target": target,
                "video_id": video_id,
                "hospital_id": str(group["hospital_id"].iloc[0]),
                "positive_event_count": positive_count,
                "negative_event_count": negative_count,
                "masked_event_count": masked_event_count,
                "masked_frame_count": masked_event_count * NUM_FACE_FRAMES,
                "action": "target_label_set_to_missing_before_split",
            })
    columns = [
        "issue_type",
        "target",
        "video_id",
        "hospital_id",
        "positive_event_count",
        "negative_event_count",
        "masked_event_count",
        "masked_frame_count",
        "action",
    ]
    return result, pd.DataFrame(audit_rows, columns=columns)


def _write_data_quality_report(output_dir, conflict_audit, before_manifest, after_manifest):
    audit_path = os.path.join(output_dir, "hemoglobin_conflicting_videos.csv")
    conflict_audit.to_csv(audit_path, index=False)
    target_counts = {}
    for target in ("hemoglobin_low", "hemoglobin_moderate_low"):
        target_audit = conflict_audit[conflict_audit["target"].eq(target)]
        target_counts[target] = {
            "conflicting_videos": int(target_audit["video_id"].nunique()),
            "masked_base_event_labels": int(target_audit["masked_event_count"].sum()),
            "masked_expanded_frame_labels": int(target_audit["masked_frame_count"].sum()),
            "labels_before_masking": int(
                pd.to_numeric(before_manifest[target], errors="coerce").notna().sum()
            ),
            "labels_after_masking": int(
                pd.to_numeric(after_manifest[target], errors="coerce").notna().sum()
            ),
        }
    report = {
        "schema_version": 1,
        "experiment": "exp2_face_only_aug20_24h_native_resolution",
        "lab_report_time": {
            "source_column": "报告时间",
            "source_timezone": LAB_REPORT_TIMEZONE,
            "stored_representation": "UTC Unix seconds",
            "previous_issue": (
                "timezone-naive Chinese local times were interpreted as UTC, "
                "shifting labels by 8 hours"
            ),
            "change": (
                "localize to Asia/Shanghai before conversion to UTC Unix seconds"
            ),
        },
        "hemoglobin_conflicting_video_policy": {
            "issue": (
                "one physical video can receive both class labels from different "
                "lab report events"
            ),
            "scope": "per target",
            "targets": ["hemoglobin_low", "hemoglobin_moderate_low"],
            "change": (
                "set the affected target label to missing for every event from that "
                "video before patient splitting"
            ),
            "effect": (
                "the video is excluded from train, validation, and test for the "
                "affected target while non-conflicting targets remain usable"
            ),
            "details_file": os.path.basename(audit_path),
            "counts": target_counts,
        },
    }
    report_path = os.path.join(output_dir, "data_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def build_features(output_dir=OUTPUT_DIR, max_samples=None):
    _ensure_dirs(output_dir)
    print("=" * 68)
    print("Exp2 Aug20 Native-Resolution Non-Adjacent RGB Dataset Builder")
    print("=" * 68)

    print("\n[1/4] Reading corrected lab and patient metadata ...")
    lab_df = _read_lab_csv(LAB_CSV)
    lab_timeseries = _build_lab_timeseries_from_df(lab_df)
    lab_metadata = _build_lab_metadata_from_df(lab_df)
    info_lookup = _read_merged_patient_info()
    lab_timeseries.to_csv(os.path.join(output_dir, "lab_timeseries.csv"), index=False)
    print(
        f"  {len(lab_timeseries)} measurements, "
        f"{lab_timeseries[['hospital_id', 'timestamp_unix']].drop_duplicates().shape[0]} lab events"
    )

    print("\n[2/4] Collapsing cleaned sessions to physical video time points ...")
    video_table, video_skips = _build_video_table(info_lookup)
    for key, value in video_skips.items():
        print(f"  {key}: {value}")
    print(
        f"  -> {len(video_table)} videos, "
        f"{video_table['hospital_id'].nunique() if len(video_table) else 0} patients"
    )

    print("\n[3/4] Matching each lab event to the nearest video ...")
    manifest, event_skips = _build_event_manifest(video_table, lab_timeseries, lab_metadata)
    for key, value in event_skips.items():
        print(f"  {key}: {value}")
    if max_samples is not None and len(manifest) > max_samples:
        rng = np.random.default_rng(SEED)
        chosen = np.sort(rng.choice(len(manifest), size=max_samples, replace=False))
        manifest = manifest.iloc[chosen].reset_index(drop=True)
    print(
        f"  -> {len(manifest)} events "
        f"({int((manifest['event_type'] == 'lab').sum())} lab, "
        f"{int((manifest['event_type'] == 'blood_pressure').sum())} BP)"
    )
    print(f"  -> max allowed lab-video delta: {LAB_MATCH_MAX_DELTA_HOURS:g} hours")

    print("\n[4/4] Extracting 20 native-resolution non-adjacent RGB frames ...")
    referenced_ids = set(manifest["video_id"].astype(str))
    referenced_videos = video_table[
        video_table["video_id"].astype(str).isin(referenced_ids)
    ].sort_values("video_id")
    face_groups = []
    successful_video_ids = []
    frame_failures = []
    for count, video in enumerate(referenced_videos.itertuples(index=False), start=1):
        try:
            frames = _load_rgb_frame_group(
                video.video_path, video.video_id, output_dir
            )
        except Exception as exc:
            frame_failures.append({"video_id": video.video_id, "error": str(exc)})
            continue
        face_groups.append(frames)
        successful_video_ids.append(video.video_id)
        if count % 100 == 0:
            print(f"  processed {count}/{len(referenced_videos)} videos")

    video_index = {video_id: index for index, video_id in enumerate(successful_video_ids)}
    base_manifest = manifest[manifest["video_id"].isin(video_index)].copy()
    base_manifest["video_index"] = base_manifest["video_id"].map(video_index).astype(int)
    base_manifest = base_manifest.sort_values(
        ["hospital_id", "label_time_unix", "sample_id"]
    ).reset_index(drop=True)
    labels_before_conflict_masking = base_manifest.copy()
    base_manifest, conflict_audit = _mask_conflicting_hemoglobin_video_labels(
        base_manifest
    )
    _write_data_quality_report(
        output_dir,
        conflict_audit,
        labels_before_conflict_masking,
        base_manifest,
    )
    print(
        f"  -> masked {int(conflict_audit['masked_event_count'].sum())} "
        f"Hb event labels from {conflict_audit['video_id'].nunique()} conflicting videos"
    )
    manifest = _expand_event_frames(base_manifest)
    manifest = manifest.sort_values(
        ["hospital_id", "label_time_unix", "base_event_id", "frame_index"]
    ).reset_index(drop=True)
    face = (
        np.stack(face_groups, axis=0)
        if face_groups
        else np.empty((0, NUM_FACE_FRAMES, 3, FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    )

    if frame_failures:
        pd.DataFrame(frame_failures).to_csv(
            os.path.join(output_dir, "frame_failures.csv"), index=False
        )
        print(f"  frame failures: {len(frame_failures)}")
    print(f"  -> RGB tensor {face.shape}, dtype={face.dtype}")
    print(
        f"  -> final {len(base_manifest)} base events expanded to "
        f"{len(manifest)} frame samples using {len(face)} unique videos"
    )

    audit = referenced_videos.drop(columns=["video_path"]).copy()
    audit["video_index"] = audit["video_id"].map(video_index)
    audit["frame_load_ok"] = audit["video_id"].isin(video_index).astype(int)
    audit.to_csv(os.path.join(output_dir, "video_summary.csv"), index=False)
    base_manifest.to_csv(os.path.join(output_dir, "base_manifest.csv"), index=False)
    manifest.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    _label_summary(manifest).to_csv(
        os.path.join(output_dir, "label_summary.csv"), index=False
    )
    np.savez_compressed(
        os.path.join(output_dir, "features.npz"),
        video_id=np.asarray(successful_video_ids, dtype=str),
        face=face,
        targets=np.asarray(TARGETS, dtype=str),
    )
    print(f"Done. Outputs saved to {output_dir}/")
    return manifest, face


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Aug20 non-adjacent RGB Exp2 features")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    arguments = parser.parse_args()
    build_features(output_dir=arguments.output_dir, max_samples=arguments.max_samples)
