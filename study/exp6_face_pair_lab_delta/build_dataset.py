"""Build chronological same-patient face pairs and leakage-free splits."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from study.exp2_face_history_head32_regression import source_data
from study.exp2_face_history_head32_regression.frame_index import (
    build_or_reuse_frame_index,
)

from .config import (
    CACHE_DIR,
    OUTPUT_DIR,
    SEED,
    SPLIT_CANDIDATES,
    SPLIT_FRACTIONS,
    SPLIT_QUANTILE_BINS,
    TARGET_ANALYTES,
    TARGETS,
    TARGET_UNITS,
)


BILIRUBIN_DEFINITION = {
    "value_column": "total_bilirubin_value",
    "direction": "high",
    "threshold": 21.0,
    "scale": 10.0,
    "unit": "umol/L",
}


def _register_bilirubin():
    source_data.TARGET_ANALYTES["total_bilirubin_high"] = "total_bilirubin"
    source_data.SCORE_DEFINITIONS["total_bilirubin_high"] = BILIRUBIN_DEFINITION


def _interval_distance(timestamp, start, end):
    if timestamp < start:
        return start - timestamp
    if timestamp > end:
        return timestamp - end
    return 0.0


def _pair_target(base, target):
    analyte = TARGET_ANALYTES[target]
    value_column = f"{analyte}_value"
    time_column = f"{analyte}_lab_time_unix"
    delta_column = f"{analyte}_delta_h"
    events = base.loc[
        pd.to_numeric(base[value_column], errors="coerce").notna()
        & pd.to_numeric(base[time_column], errors="coerce").notna()
    ].copy()
    events["lab_value"] = pd.to_numeric(events[value_column], errors="raise")
    events["lab_time_unix"] = pd.to_numeric(events[time_column], errors="raise")
    events["match_delta_h"] = pd.to_numeric(events[delta_column], errors="raise")
    events["video_time_unix"] = pd.to_numeric(events["capture_time_unix"], errors="raise")

    # One face video per actual laboratory event prevents duplicated labels from
    # videos that all matched the same measurement.
    events = events.sort_values(
        ["hospital_id", "lab_time_unix", "match_delta_h", "video_time_unix", "video_id"]
    ).drop_duplicates(["hospital_id", "lab_time_unix"], keep="first")

    rows = []
    exclusions = {
        "patients_with_fewer_than_two_unique_events": 0,
        "non_increasing_video_time": 0,
        "same_video": 0,
    }
    for hospital_id, group in events.groupby("hospital_id", sort=True):
        group = group.sort_values(["lab_time_unix", "video_time_unix", "video_id"])
        if len(group) < 2:
            exclusions["patients_with_fewer_than_two_unique_events"] += 1
            continue
        values = list(group.itertuples(index=False))
        for first, second in zip(values[:-1], values[1:]):
            if str(first.video_id) == str(second.video_id):
                exclusions["same_video"] += 1
                continue
            if float(second.video_time_unix) <= float(first.video_time_unix):
                exclusions["non_increasing_video_time"] += 1
                continue
            rows.append({
                "pair_id": (
                    f"{target}:{hospital_id}:{int(first.lab_time_unix)}:"
                    f"{int(second.lab_time_unix)}"
                ),
                "target": target,
                "analyte": analyte,
                "unit": TARGET_UNITS[target],
                "hospital_id": str(hospital_id),
                "first_video_id": str(first.video_id),
                "second_video_id": str(second.video_id),
                "first_lab_time_unix": float(first.lab_time_unix),
                "second_lab_time_unix": float(second.lab_time_unix),
                "first_video_time_unix": float(first.video_time_unix),
                "second_video_time_unix": float(second.video_time_unix),
                "first_match_delta_h": float(first.match_delta_h),
                "second_match_delta_h": float(second.match_delta_h),
                "first_value": float(first.lab_value),
                "second_value": float(second.lab_value),
                "raw_delta": float(second.lab_value - first.lab_value),
                "lab_interval_h": float(
                    (second.lab_time_unix - first.lab_time_unix) / 3600.0
                ),
                "video_interval_h": float(
                    (second.video_time_unix - first.video_time_unix) / 3600.0
                ),
            })
    records = pd.DataFrame(rows)
    if records.empty:
        raise RuntimeError(f"No chronological pairs for {target}")
    if records["pair_id"].duplicated().any():
        raise AssertionError(f"Duplicate pair IDs for {target}")
    return records, {
        "target": target,
        "matched_unique_events": int(len(events)),
        "matched_patients": int(events.hospital_id.nunique()),
        "candidate_pairs": int(len(records)),
        **exclusions,
    }


def _stratified_candidate(patient_table, rng):
    quantiles = min(SPLIT_QUANTILE_BINS, len(patient_table))
    bins = pd.qcut(
        patient_table["median_delta"], q=quantiles, labels=False, duplicates="drop"
    )
    assignments = {}
    for _, group in patient_table.assign(_bin=bins).groupby("_bin", dropna=False):
        ids = group["hospital_id"].to_numpy(str)
        rng.shuffle(ids)
        count = len(ids)
        n_train = int(round(count * SPLIT_FRACTIONS[0]))
        n_val = int(round(count * SPLIT_FRACTIONS[1]))
        if count >= 3:
            n_train = min(max(n_train, 1), count - 2)
            n_val = min(max(n_val, 1), count - n_train - 1)
        for patient in ids[:n_train]:
            assignments[patient] = "train"
        for patient in ids[n_train:n_train + n_val]:
            assignments[patient] = "val"
        for patient in ids[n_train + n_val:]:
            assignments[patient] = "test"
    return assignments


def _score_split(records, assignments):
    split = records.hospital_id.map(assignments)
    if split.isna().any() or set(split) != {"train", "val", "test"}:
        return np.inf, {}
    values = records.raw_delta.to_numpy(float)
    global_iqr = max(float(np.subtract(*np.quantile(values, [0.75, 0.25]))), 1e-9)
    pair_metrics = []
    for first, second in combinations(("train", "val", "test"), 2):
        x = values[split.eq(first).to_numpy()]
        y = values[split.eq(second).to_numpy()]
        if len(x) < 2 or len(y) < 2:
            return np.inf, {}
        pair_metrics.append({
            "first": first,
            "second": second,
            "ks": float(ks_2samp(x, y).statistic),
            "wasserstein_iqr": float(wasserstein_distance(x, y) / global_iqr),
        })
    fractions = split.value_counts(normalize=True)
    size_error = sum(
        abs(float(fractions.get(name, 0.0)) - wanted)
        for name, wanted in zip(("train", "val", "test"), SPLIT_FRACTIONS)
    )
    positive_rates = [
        float((records.loc[split.eq(name), "raw_delta"] > 0).mean())
        for name in ("train", "val", "test")
    ]
    score = (
        max(item["ks"] for item in pair_metrics)
        + max(item["wasserstein_iqr"] for item in pair_metrics)
        + size_error
        + (max(positive_rates) - min(positive_rates))
    )
    return score, {
        "pairwise": pair_metrics,
        "size_error": size_error,
        "positive_rate_range": max(positive_rates) - min(positive_rates),
    }


def _add_split_and_scaling(records, target, seed):
    patient_table = records.groupby("hospital_id", as_index=False).agg(
        median_delta=("raw_delta", "median"), pair_count=("pair_id", "size")
    )
    if len(patient_table) < 9:
        raise RuntimeError(f"Too few paired patients for {target}: {len(patient_table)}")
    best = None
    for candidate in range(SPLIT_CANDIDATES):
        rng = np.random.default_rng(seed + candidate * 104729)
        assignments = _stratified_candidate(patient_table, rng)
        score, details = _score_split(records, assignments)
        if best is None or score < best[0]:
            best = (score, candidate, assignments, details)
    score, candidate, assignments, details = best
    result = records.copy()
    result["split"] = result.hospital_id.map(assignments)
    if result.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError(f"Patient leakage for {target}")
    train = result.loc[result.split.eq("train"), "raw_delta"].to_numpy(float)
    median = float(np.median(train))
    q1, q3 = np.quantile(train, [0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        raise RuntimeError(f"Degenerate train delta IQR for {target}: {iqr}")
    result["scaled_delta"] = (result.raw_delta - median) / iqr
    audit_rows = []
    for split, group in result.groupby("split", sort=False):
        audit_rows.append({
            "target": target,
            "split": split,
            "patients": int(group.hospital_id.nunique()),
            "pairs": int(len(group)),
            "positive_fraction": float(group.raw_delta.gt(0).mean()),
            "delta_mean": float(group.raw_delta.mean()),
            "delta_std": float(group.raw_delta.std()),
            "delta_median": float(group.raw_delta.median()),
            "delta_q25": float(group.raw_delta.quantile(0.25)),
            "delta_q75": float(group.raw_delta.quantile(0.75)),
        })
    scaler = {
        "target": target,
        "unit": TARGET_UNITS[target],
        "fit_split": "train",
        "median": median,
        "iqr": iqr,
    }
    selection = {
        "candidate_count": SPLIT_CANDIDATES,
        "selected_candidate": candidate,
        "objective": float(score),
        **details,
    }
    return result, scaler, audit_rows, selection


def prepare(targets=TARGETS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUTPUT_DIR / "source_data"
    task_dir = OUTPUT_DIR / "task_records"
    task_dir.mkdir(parents=True, exist_ok=True)
    _register_bilirubin()
    base, videos, quality = source_data.build_raw_video_source(
        str(source_dir), tuple(targets)
    )
    candidates, summaries = {}, []
    for target in targets:
        candidates[target], summary = _pair_target(base, target)
        summaries.append(summary)
    required_ids = sorted({
        video_id
        for records in candidates.values()
        for field in ("first_video_id", "second_video_id")
        for video_id in records[field].astype(str)
    })
    index_rows = videos.loc[videos.video_id.astype(str).isin(required_ids)]
    frame_index = build_or_reuse_frame_index(index_rows, CACHE_DIR, "20frame")
    usable = set(frame_index.video_lookup)

    split_rows, scalers, selections, final_records = [], {}, {}, {}
    for target, records in candidates.items():
        valid = records.first_video_id.isin(usable) & records.second_video_id.isin(usable)
        summaries_by_target = next(row for row in summaries if row["target"] == target)
        summaries_by_target["pairs_excluded_invalid_frames"] = int((~valid).sum())
        records = records.loc[valid].reset_index(drop=True)
        records, scaler, audit, selection = _add_split_and_scaling(
            records, target, SEED
        )
        records.to_csv(task_dir / f"{target}.csv", index=False)
        final_records[target] = records
        scalers[target] = scaler
        selections[target] = selection
        split_rows.extend(audit)
        summaries_by_target.update({
            "usable_pairs": int(len(records)),
            "usable_patients": int(records.hospital_id.nunique()),
            "usable_videos": int(pd.unique(pd.concat([
                records.first_video_id, records.second_video_id
            ])).size),
        })

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "task_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUTPUT_DIR / "split_distribution.csv", index=False)
    (OUTPUT_DIR / "target_scalers.json").write_text(
        json.dumps(scalers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "exp6_paired_face_lab_delta_regression",
        "targets": list(targets),
        "target_analytes": {target: TARGET_ANALYTES[target] for target in targets},
        "label": "second canonical lab value minus first canonical lab value",
        "pair_policy": (
            "each raw video receives its nearest lab within 24 hours; when several "
            "videos map to one lab event, retain the closest video; pair consecutive "
            "unique events within patient with distinct videos and increasing lab and "
            "video times"
        ),
        "split_policy": (
            "patient-disjoint 60/20/20; 512 stratified candidates selected by delta "
            "KS, normalized Wasserstein, split size, and direction-rate agreement"
        ),
        "scaling": "train-only median/IQR of raw delta",
        "frames": "20 deterministic nonadjacent source frames per video",
        "views": ["original", "hflip", "center_crop", "brightness", "contrast"],
        "source_quality": quality,
        "split_selections": selections,
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[prepare-complete] tasks={len(final_records)} "
        f"pairs={sum(map(len, final_records.values()))} "
        f"indexed_videos={len(frame_index.video_ids)}",
        flush=True,
    )
    return final_records, scalers, frame_index
