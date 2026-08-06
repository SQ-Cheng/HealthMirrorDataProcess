"""Build auditable postoperative recovery labels without using lab values."""

from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from study.exp2_lab_multimodal.build_dataset import (
    _normalize_hospital_id,
    _read_merged_patient_info,
)

from .config import (
    DATA_ROOT,
    LAB_METADATA_CSV,
    MAX_TIME_SOURCE_DELTA_SECONDS,
    SEED,
    SPLIT_CANDIDATES,
    SPLIT_FRACTIONS,
    SPLIT_SCORE_BINS,
    TIMEZONE,
)


METADATA_COLUMNS = (
    "首页病案号",
    "首页入院时间",
    "首页出院时间",
    "手术开始日期",
    "手术结束日期",
    "首页手术操作名称",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(value):
    return [token.strip() for token in str(value).split("^")]


def load_surgical_episodes():
    raw = pd.read_csv(
        LAB_METADATA_CSV,
        dtype=str,
        keep_default_na=False,
        usecols=list(METADATA_COLUMNS),
    ).drop_duplicates()
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["admission_time"] = pd.to_datetime(raw["首页入院时间"], errors="coerce")
    raw["discharge_time"] = pd.to_datetime(raw["首页出院时间"], errors="coerce")
    episode_rows, event_rows = [], []
    for _, episode in raw.iterrows():
        starts = _tokens(episode["手术开始日期"])
        ends = _tokens(episode["手术结束日期"])
        names = _tokens(episode["首页手术操作名称"])
        valid_events = []
        for position in range(max(len(starts), len(ends), len(names))):
            start_text = starts[position] if position < len(starts) else ""
            end_text = ends[position] if position < len(ends) else ""
            name = names[position] if position < len(names) else ""
            start = pd.to_datetime(start_text, errors="coerce")
            end = pd.to_datetime(end_text, errors="coerce")
            parsed = pd.notna(start) and pd.notna(end)
            valid = bool(
                episode.hospital_id
                and parsed
                and pd.notna(episode.admission_time)
                and pd.notna(episode.discharge_time)
                and end > start
                and start >= episode.admission_time
                and end <= episode.discharge_time
            )
            event_rows.append({
                "hospital_id": episode.hospital_id,
                "admission_time": episode.admission_time,
                "discharge_time": episode.discharge_time,
                "event_position": position,
                "surgery_name": name,
                "surgery_start": start,
                "surgery_end": end,
                "valid_event": valid,
                "token_counts_match": len(starts) == len(ends) == len(names),
            })
            if valid:
                valid_events.append((start, end, name, position))
        if not valid_events:
            continue
        # Recovery begins after the final valid operation in the hospitalization.
        start, end, name, position = max(valid_events, key=lambda event: event[1])
        episode_rows.append({
            "hospital_id": episode.hospital_id,
            "admission_time": episode.admission_time,
            "discharge_time": episode.discharge_time,
            "index_surgery_start": start,
            "index_surgery_end": end,
            "index_surgery_name": name,
            "index_surgery_position": position,
            "valid_surgery_count": len(valid_events),
        })
    episodes = pd.DataFrame(episode_rows).drop_duplicates(
        ["hospital_id", "admission_time", "discharge_time"]
    )
    events = pd.DataFrame(event_rows)
    return episodes.sort_values(
        ["hospital_id", "admission_time", "discharge_time"]
    ).reset_index(drop=True), events


def _video_bounds(timestamp_path):
    values, invalid, nonmonotonic = [], 0, 0
    previous = None
    try:
        with open(timestamp_path, encoding="utf-8", errors="replace") as handle:
            next(handle, None)
            for line in handle:
                try:
                    value = float(line.rsplit(",", 1)[-1].strip())
                except (IndexError, ValueError):
                    invalid += 1
                    continue
                if not np.isfinite(value):
                    invalid += 1
                    continue
                if previous is not None and value < previous:
                    nonmonotonic += 1
                previous = value
                values.append(value)
    except OSError:
        return None
    if not values:
        return None
    return {
        "capture_start_unix": float(min(values)),
        "capture_end_unix": float(max(values)),
        "timestamp_rows": len(values),
        "invalid_timestamp_rows": invalid,
        "nonmonotonic_steps": nonmonotonic,
    }


def _session_timestamp(patient_info_path):
    try:
        text = Path(patient_info_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return pd.NaT
    match = re.search(r"^Session Timestamp:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return pd.to_datetime(match.group(1), errors="coerce") if match else pd.NaT


def _unix_to_local_naive(value):
    return pd.Timestamp(value, unit="s", tz="UTC").tz_convert(TIMEZONE).tz_localize(None)


def build_recovery_candidates(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, surgery_events = load_surgical_episodes()
    episodes_by_patient = {
        hospital_id: group for hospital_id, group in episodes.groupby("hospital_id")
    }
    mappings = _read_merged_patient_info()
    video_paths = sorted(glob.glob(str(DATA_ROOT / "mirror*_data/patient_*/video.avi")))
    records, audit_rows = [], []
    exclusions = Counter()
    for video_path_text in video_paths:
        video_path = Path(video_path_text)
        match = re.search(r"/(mirror\d+)_data/patient_(\d+)/video\.avi$", video_path_text)
        if match is None:
            exclusions["path_parse_failed"] += 1
            continue
        mirror, local_id = match.group(1), int(match.group(2))
        video_id = f"{mirror}_patient_{local_id:06d}"
        mapping = mappings.get((mirror, local_id))
        hospital_id = _normalize_hospital_id(
            mapping.get("Hospital_Patient_ID", "") if mapping else ""
        )
        bounds = _video_bounds(str(video_path) + ".ts")
        session_time = _session_timestamp(video_path.with_name("patient_info.txt"))
        row = {
            "video_id": video_id,
            "mirror": mirror,
            "lab_patient_id": local_id,
            "hospital_id": hospital_id,
            "video_path": str(video_path),
            "session_time": session_time,
        }
        if not hospital_id:
            status = "invalid_or_missing_patient_mapping"
        elif bounds is None:
            status = "missing_or_invalid_video_timestamps"
        elif pd.isna(session_time):
            status = "missing_or_invalid_session_timestamp"
        else:
            row.update(bounds)
            capture_start = _unix_to_local_naive(bounds["capture_start_unix"])
            capture_end = _unix_to_local_naive(bounds["capture_end_unix"])
            capture_midpoint = _unix_to_local_naive(
                (bounds["capture_start_unix"] + bounds["capture_end_unix"]) / 2.0
            )
            source_delta = abs((session_time - capture_start).total_seconds())
            row.update({
                "capture_start_local": capture_start,
                "capture_end_local": capture_end,
                "capture_midpoint_local": capture_midpoint,
                "time_source_delta_seconds": source_delta,
            })
            if source_delta > MAX_TIME_SOURCE_DELTA_SECONDS:
                status = "time_sources_disagree_gt_5min"
            elif hospital_id not in episodes_by_patient:
                status = "no_valid_surgery_episode"
            else:
                candidates = episodes_by_patient[hospital_id]
                matched = candidates.loc[
                    candidates["index_surgery_end"].le(capture_start)
                    & candidates["discharge_time"].ge(capture_end)
                ]
                if matched.empty:
                    status = "video_outside_postop_to_discharge"
                elif len(matched) > 1:
                    status = "ambiguous_hospitalization_episode"
                else:
                    episode = matched.iloc[0]
                    denominator = episode.discharge_time - episode.index_surgery_end
                    recovery = (capture_midpoint - episode.index_surgery_end) / denominator
                    if not (0.0 <= recovery <= 1.0):
                        status = "computed_recovery_outside_0_1"
                    else:
                        status = "retained"
                        records.append({
                            **row,
                            "admission_time": episode.admission_time,
                            "discharge_time": episode.discharge_time,
                            "index_surgery_start": episode.index_surgery_start,
                            "index_surgery_end": episode.index_surgery_end,
                            "index_surgery_name": episode.index_surgery_name,
                            "valid_surgery_count": int(episode.valid_surgery_count),
                            "postoperative_duration_hours": denominator.total_seconds() / 3600.0,
                            "hours_after_surgery": (
                                capture_midpoint - episode.index_surgery_end
                            ).total_seconds() / 3600.0,
                            "recovery_score": float(recovery),
                        })
        exclusions[status] += 1
        audit_rows.append({**row, "status": status})
    records = pd.DataFrame(records).sort_values("video_id").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("video_id").reset_index(drop=True)
    if records.empty or records["video_id"].duplicated().any():
        raise RuntimeError("Recovery candidate construction failed or produced duplicate videos")
    records.to_csv(output_dir / "recovery_candidates.csv", index=False)
    audit.to_csv(output_dir / "video_eligibility_audit.csv", index=False)
    episodes.to_csv(output_dir / "surgical_episodes.csv", index=False)
    surgery_events.to_csv(output_dir / "surgery_event_audit.csv", index=False)
    report = {
        "schema_version": 1,
        "experiment": "exp4_postoperative_recovery_from_face",
        "label_definition": {
            "zero_time": "end of final valid surgery in hospitalization",
            "one_time": "hospital discharge",
            "interpolation": "linear at video capture interval midpoint",
            "video_interval_requirement": "entire interval within [surgery_end, discharge]",
        },
        "time_policy": {
            "timezone": TIMEZONE,
            "primary": "video.avi.ts capture interval",
            "validation": "patient_info.txt Session Timestamp compared with capture start",
            "maximum_absolute_source_delta_seconds": MAX_TIME_SOURCE_DELTA_SECONDS,
        },
        "counts": {
            "raw_video_files": len(video_paths),
            "valid_surgical_episodes": len(episodes),
            "surgical_patients": int(episodes["hospital_id"].nunique()),
            "retained_videos_before_frame_validation": len(records),
            "retained_patients_before_frame_validation": int(records["hospital_id"].nunique()),
            "statuses": dict(exclusions),
        },
        "source": {
            "path": str(LAB_METADATA_CSV),
            "sha256": _sha256(LAB_METADATA_CSV),
            "columns_used": list(METADATA_COLUMNS),
            "lab_result_values_used": False,
        },
    }
    with open(output_dir / "data_quality_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"[data] retained={len(records)} patients={records.hospital_id.nunique()} "
        f"statuses={dict(exclusions)}",
        flush=True,
    )
    return records, report


def _candidate_split(records, rng):
    patients = records.groupby("hospital_id").agg(
        median_score=("recovery_score", "median"),
        video_count=("video_id", "size"),
    ).reset_index()
    quantiles = min(SPLIT_SCORE_BINS, max(3, len(patients) // 12))
    patients["stratum"] = pd.qcut(
        patients["median_score"].rank(method="first"), quantiles, labels=False
    )
    assignment = {}
    for _, group in patients.groupby("stratum"):
        ids = group["hospital_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        train_n = max(1, int(round(n * SPLIT_FRACTIONS[0])))
        val_n = max(1, int(round(n * SPLIT_FRACTIONS[1])))
        if train_n + val_n >= n:
            train_n, val_n = n - 2, 1
        for hospital_id in ids[:train_n]:
            assignment[hospital_id] = "train"
        for hospital_id in ids[train_n:train_n + val_n]:
            assignment[hospital_id] = "val"
        for hospital_id in ids[train_n + val_n:]:
            assignment[hospital_id] = "test"
    return records["hospital_id"].map(assignment)


def add_balanced_patient_split(records, output_dir, seed=SEED):
    values = records["recovery_score"].to_numpy(float)
    global_iqr = max(float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-9)
    best = None
    for candidate in range(SPLIT_CANDIDATES):
        split = _candidate_split(records, np.random.default_rng(seed + candidate))
        if split.isna().any() or set(split) != {"train", "val", "test"}:
            continue
        pair_stats = []
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            a = values[split.eq(first)]; b = values[split.eq(second)]
            pair_stats.append((
                float(ks_2samp(a, b).statistic),
                float(wasserstein_distance(a, b) / global_iqr),
            ))
        video_fractions = np.array([split.eq(name).mean() for name in ("train", "val", "test")])
        patient_table = records[["hospital_id"]].drop_duplicates().copy()
        patient_table["split"] = patient_table["hospital_id"].map(
            records.assign(split=split).drop_duplicates("hospital_id").set_index("hospital_id")["split"]
        )
        patient_fractions = np.array([
            patient_table["split"].eq(name).mean() for name in ("train", "val", "test")
        ])
        size_error = max(
            np.abs(video_fractions - SPLIT_FRACTIONS).max(),
            np.abs(patient_fractions - SPLIT_FRACTIONS).max(),
        )
        objective = 2 * max(row[1] for row in pair_stats) + max(row[0] for row in pair_stats) + size_error
        if best is None or objective < best[0]:
            best = (objective, candidate, split.copy(), pair_stats, size_error)
    if best is None:
        raise RuntimeError("Could not construct patient-disjoint split")
    _, candidate, split, pair_stats, size_error = best
    result = records.copy()
    result["split"] = split
    if result.groupby("hospital_id")["split"].nunique().gt(1).any():
        raise AssertionError("Patient leakage in Exp4 split")
    output_dir = Path(output_dir)
    result.to_csv(output_dir / "records.csv", index=False)
    rows = []
    for name in ("train", "val", "test"):
        selected = result.loc[result["split"].eq(name), "recovery_score"]
        rows.append({
            "split": name,
            "videos": len(selected),
            "patients": int(result.loc[result["split"].eq(name), "hospital_id"].nunique()),
            "mean": selected.mean(),
            "std": selected.std(),
            "min": selected.min(),
            "q25": selected.quantile(0.25),
            "median": selected.median(),
            "q75": selected.quantile(0.75),
            "max": selected.max(),
        })
    pd.DataFrame(rows).to_csv(output_dir / "split_distribution.csv", index=False)
    manifest = {
        "schema_version": 1,
        "algorithm": "patient-level median-score stratification with candidate search",
        "seed": seed,
        "candidate_count": SPLIT_CANDIDATES,
        "selected_candidate": candidate,
        "target_fractions": dict(zip(("train", "val", "test"), SPLIT_FRACTIONS)),
        "objective": best[0],
        "maximum_size_fraction_error": size_error,
        "pairwise_ks_wasserstein_iqr": {
            pair: {"ks": stats[0], "wasserstein_iqr": stats[1]}
            for pair, stats in zip(("train_val", "train_test", "val_test"), pair_stats)
        },
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        "[split] " + " ".join(
            f"{name}={int(result.split.eq(name).sum())}videos/"
            f"{result.loc[result.split.eq(name), 'hospital_id'].nunique()}patients"
            for name in ("train", "val", "test")
        ),
        flush=True,
    )
    return result, manifest
