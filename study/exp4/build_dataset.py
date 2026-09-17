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
    SPLIT_KS_MAX,
    SPLIT_SCORE_BINS,
    SPLIT_SIZE_FRACTION_MAX,
    SPLIT_SMALL_KS_MAX,
    SPLIT_SMALL_N,
    SPLIT_SMALL_WASSERSTEIN_IQR_MAX,
    SPLIT_WASSERSTEIN_IQR_MAX,
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


def _split_allocation(count):
    if count < 3:
        raise ValueError(f"At least three patients per stratum are required, got {count}")
    train = max(1, int(round(count * SPLIT_FRACTIONS[0])))
    validation = max(1, int(round(count * SPLIT_FRACTIONS[1])))
    if train + validation > count - 1:
        train = count - validation - 1
    return train, validation, count - train - validation


def _patient_strata(records):
    patients = records.groupby("hospital_id").agg(
        median_score=("recovery_score", "median"),
        video_count=("video_id", "size"),
    ).reset_index().sort_values("hospital_id").reset_index(drop=True)
    quantiles = min(SPLIT_SCORE_BINS, max(3, len(patients) // 12))
    patients["stratum"] = pd.qcut(
        patients["median_score"].rank(method="first"), quantiles, labels=False
    )
    if patients["stratum"].value_counts().min() < 3:
        raise RuntimeError("Recovery-score stratum has fewer than three patients")
    return patients


def _candidate_assignment(patients, rng):
    assignment = np.empty(len(patients), dtype=np.int8)
    strata = patients["stratum"].to_numpy(np.int16)
    for stratum in sorted(np.unique(strata)):
        indices = rng.permutation(np.flatnonzero(strata == stratum))
        train_count, validation_count, _ = _split_allocation(len(indices))
        assignment[indices[:train_count]] = 0
        assignment[indices[train_count:train_count + validation_count]] = 1
        assignment[indices[train_count + validation_count:]] = 2
    return assignment


def _distribution_pair(values, split_codes, first, second):
    first_values = values[split_codes == first]
    second_values = values[split_codes == second]
    global_iqr = max(
        float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-9
    )
    ks = float(ks_2samp(first_values, second_values).statistic)
    wasserstein = float(wasserstein_distance(first_values, second_values))
    wasserstein_iqr = wasserstein / global_iqr
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    quantile_difference = float(
        np.max(
            np.abs(
                np.quantile(first_values, quantiles)
                - np.quantile(second_values, quantiles)
            )
        )
        / global_iqr
    )
    small = min(len(first_values), len(second_values)) < SPLIT_SMALL_N
    ks_limit = SPLIT_SMALL_KS_MAX if small else SPLIT_KS_MAX
    wasserstein_limit = (
        SPLIT_SMALL_WASSERSTEIN_IQR_MAX
        if small
        else SPLIT_WASSERSTEIN_IQR_MAX
    )
    return {
        "n_first": int(len(first_values)),
        "n_second": int(len(second_values)),
        "ks": ks,
        "wasserstein": wasserstein,
        "global_iqr": global_iqr,
        "wasserstein_iqr": wasserstein_iqr,
        "max_quantile_difference_iqr": quantile_difference,
        "small_sample_rule": bool(small),
        "ks_limit": ks_limit,
        "wasserstein_iqr_limit": wasserstein_limit,
        "passed": bool(
            ks <= ks_limit + 1e-12
            and wasserstein_iqr <= wasserstein_limit + 1e-12
        ),
    }


def _candidate_score(values, patient_assignment, patient_row_indices):
    split_codes = patient_assignment[patient_row_indices]
    pair_metrics = [
        _distribution_pair(values, split_codes, first, second)
        for first, second in ((0, 1), (0, 2), (1, 2))
    ]
    max_ks = max(row["ks"] for row in pair_metrics)
    max_wasserstein = max(row["wasserstein_iqr"] for row in pair_metrics)
    max_quantile = max(row["max_quantile_difference_iqr"] for row in pair_metrics)
    video_fractions = np.asarray([
        (split_codes == code).mean() for code in range(3)
    ])
    patient_fractions = np.asarray([
        (patient_assignment == code).mean() for code in range(3)
    ])
    target_fractions = np.asarray(SPLIT_FRACTIONS)
    size_error = float(max(
        np.abs(video_fractions - target_fractions).max(),
        np.abs(patient_fractions - target_fractions).max(),
    ))
    objective = (
        2.0 * max_wasserstein
        + max_ks
        + 0.25 * max_quantile
        + 0.25 * size_error
    )
    return {
        "objective": float(objective),
        "max_ks": float(max_ks),
        "max_wasserstein_iqr": float(max_wasserstein),
        "max_quantile_difference_iqr": float(max_quantile),
        "size_fraction_error": size_error,
        "passed": bool(
            all(row["passed"] for row in pair_metrics)
            and size_error <= SPLIT_SIZE_FRACTION_MAX + 1e-12
        ),
    }


def _distribution_audit(records):
    values = records["recovery_score"].to_numpy(np.float64)
    split_codes = records["split"].map(
        {"train": 0, "val": 1, "test": 2}
    ).to_numpy(np.int8)
    summary_rows = []
    for split in ("train", "val", "test"):
        selected = values[records["split"].eq(split)]
        split_records = records.loc[records["split"].eq(split)]
        summary_rows.append({
            "variable": "recovery_score",
            "split": split,
            "videos": int(len(selected)),
            "patients": int(split_records["hospital_id"].nunique()),
            "mean": float(np.mean(selected)),
            "std": float(np.std(selected, ddof=1)),
            "minimum": float(np.min(selected)),
            "q10": float(np.quantile(selected, 0.10)),
            "q25": float(np.quantile(selected, 0.25)),
            "median": float(np.quantile(selected, 0.50)),
            "q75": float(np.quantile(selected, 0.75)),
            "q90": float(np.quantile(selected, 0.90)),
            "maximum": float(np.max(selected)),
        })
    pair_rows = []
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        pair_rows.append({
            "variable": "recovery_score",
            "split_first": first,
            "split_second": second,
            **_distribution_pair(
                values,
                split_codes,
                {"train": 0, "val": 1, "test": 2}[first],
                {"train": 0, "val": 1, "test": 2}[second],
            ),
        })
    return summary_rows, pair_rows


def add_balanced_patient_split(records, output_dir, seed=SEED):
    records = records.copy()
    records["hospital_id"] = records["hospital_id"].astype(str)
    patients = _patient_strata(records)
    patient_lookup = {
        hospital_id: index
        for index, hospital_id in enumerate(patients["hospital_id"])
    }
    patient_row_indices = records["hospital_id"].map(patient_lookup).to_numpy(np.int64)
    values = records["recovery_score"].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    best_passed, best_overall = None, None
    for candidate_index in range(SPLIT_CANDIDATES):
        assignment = _candidate_assignment(patients, rng)
        score = _candidate_score(values, assignment, patient_row_indices)
        key = (
            score["objective"],
            score["max_wasserstein_iqr"],
            score["max_ks"],
            candidate_index,
        )
        candidate = (key, assignment.copy(), score, candidate_index)
        if best_overall is None or key < best_overall[0]:
            best_overall = candidate
        if score["passed"] and (best_passed is None or key < best_passed[0]):
            best_passed = candidate
    if best_passed is None:
        score = best_overall[2]
        raise RuntimeError(
            f"No balanced split passed after {SPLIT_CANDIDATES} candidates; "
            f"best max_KS={score['max_ks']:.4f}, "
            f"max_Wasserstein/IQR={score['max_wasserstein_iqr']:.4f}, "
            f"size_error={score['size_fraction_error']:.4f}"
        )
    _, assignment, selected_score, candidate_index = best_passed
    result = records.copy()
    result["split"] = np.asarray(("train", "val", "test"))[
        assignment[patient_row_indices]
    ]
    patient_sets = {
        split: set(result.loc[result["split"].eq(split), "hospital_id"])
        for split in ("train", "val", "test")
    }
    if any(patient_sets[first] & patient_sets[second] for first, second in (
        ("train", "val"), ("train", "test"), ("val", "test")
    )):
        raise AssertionError("Patient leakage in Exp4 split")
    if len(result) != len(records) or result["video_id"].nunique() != len(records):
        raise AssertionError("Video loss or duplication in Exp4 split")
    summary_rows, pair_rows = _distribution_audit(result)
    if not all(row["passed"] for row in pair_rows):
        raise AssertionError("Selected Exp4 split failed pairwise distribution audit")
    output_dir = Path(output_dir)
    result.to_csv(output_dir / "records.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    pairs = pd.DataFrame(pair_rows)
    summary.to_csv(output_dir / "split_distribution.csv", index=False)
    summary.to_csv(output_dir / "split_distribution_audit.csv", index=False)
    pairs.to_csv(output_dir / "split_distribution_pairwise.csv", index=False)
    manifest = {
        "schema_version": 2,
        "algorithm": "Exp2-style patient-disjoint stratified candidate search with hard video-distribution constraints",
        "seed": seed,
        "candidate_count": SPLIT_CANDIDATES,
        "selected_candidate_index": int(candidate_index),
        "score_stratification": {
            "patient_statistic": "median recovery_score",
            "quantile_bins": int(patients["stratum"].nunique()),
        },
        "target_fractions": dict(zip(("train", "val", "test"), SPLIT_FRACTIONS)),
        "hard_limits": {
            "ks": SPLIT_KS_MAX,
            "wasserstein_iqr": SPLIT_WASSERSTEIN_IQR_MAX,
            "small_n": SPLIT_SMALL_N,
            "small_ks": SPLIT_SMALL_KS_MAX,
            "small_wasserstein_iqr": SPLIT_SMALL_WASSERSTEIN_IQR_MAX,
            "size_fraction_error": SPLIT_SIZE_FRACTION_MAX,
        },
        "selection_score": selected_score,
        "all_pairwise_distribution_checks_passed": bool(pairs["passed"].all()),
        "patient_leakage": False,
        "video_count_preserved": True,
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        "[split] " + " ".join(
            f"{name}={int(result.split.eq(name).sum())}videos/"
            f"{result.loc[result.split.eq(name), 'hospital_id'].nunique()}patients"
            for name in ("train", "val", "test")
        ) + f" candidate={candidate_index} objective={selected_score['objective']:.5f}",
        flush=True,
    )
    return result, manifest
