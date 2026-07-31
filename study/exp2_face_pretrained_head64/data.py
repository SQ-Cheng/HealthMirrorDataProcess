"""Balanced binary video tasks and streaming all-frame image loading."""

from collections import OrderedDict
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from torch.utils.data import Dataset, Sampler
from torchvision.io import ImageReadMode, decode_jpeg

from .config import (
    DECODE_CACHE_FRAMES,
    FRAME_SHUFFLE_CHUNK_SIZE,
    MAX_OPEN_FILES_PER_WORKER,
    MIN_PATIENTS_PER_CLASS,
    MIN_VIDEOS_PER_CLASS,
    SCORE_DEFINITIONS,
    SEED,
    SOURCE_IMAGE_SIZE,
    SPLIT_CANDIDATES,
    SPLIT_FRACTIONS,
    SPLIT_KS_MAX,
    SPLIT_POSITIVE_RATE_RANGE_MAX,
    SPLIT_SIZE_FRACTION_MAX,
    SPLIT_SMALL_KS_MAX,
    SPLIT_SMALL_N,
    SPLIT_SMALL_WASSERSTEIN_IQR_MAX,
    SPLIT_WASSERSTEIN_IQR_MAX,
    TARGETS,
)


def validate_source_data(source_dir):
    required = ("base_manifest.csv", "data_quality_report.json", "video_summary.csv")
    missing = [name for name in required if not os.path.exists(os.path.join(source_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing corrected Exp2 source files: {missing}")
    with open(
        os.path.join(source_dir, "data_quality_report.json"), encoding="utf-8"
    ) as handle:
        quality = json.load(handle)
    timezone = quality.get("lab_report_time", {}).get("source_timezone")
    if timezone != "Asia/Shanghai":
        raise RuntimeError(f"Corrected Asia/Shanghai time basis is missing: {timezone}")
    policy = quality.get("hemoglobin_conflicting_video_policy", {})
    if policy.get("scope") != "per target" or "hemoglobin_low" not in policy.get("targets", []):
        raise RuntimeError("Corrected Hb conflict policy is missing")
    return quality


def _conflicting_video_audit(events, target):
    rows = []
    for video_id, group in events.groupby("video_id", sort=True):
        labels = pd.to_numeric(group[target], errors="coerce").dropna().astype(int)
        positive_count = int(labels.eq(1).sum())
        negative_count = int(labels.eq(0).sum())
        if positive_count and negative_count:
            rows.append({
                "target": target,
                "video_id": str(video_id),
                "hospital_id": str(group["hospital_id"].iloc[0]),
                "positive_event_count": positive_count,
                "negative_event_count": negative_count,
                "event_count": int(len(group)),
                "action": "excluded_for_this_target",
            })
    return rows


def _score_events(events, target, video_summary):
    definition = SCORE_DEFINITIONS[target]
    result = events.copy()
    result["source_sample_id"] = result["sample_id"].astype(str)
    result["score_threshold"] = np.nan
    result["score_scale"] = np.nan
    result["raw_value"] = np.nan
    result["systolic_blood_pressure"] = np.nan
    result["diastolic_blood_pressure"] = np.nan

    if target == "high_blood_pressure":
        pressure = video_summary[
            [
                "video_id",
                "high_blood_pressure",
                "low_blood_pressure",
                "blood_pressure_valid",
            ]
        ].copy()
        pressure["video_id"] = pressure["video_id"].astype(str)
        pressure = pressure.drop_duplicates("video_id", keep="first")
        result["video_id"] = result["video_id"].astype(str)
        result = result.drop(
            columns=[
                "high_blood_pressure",
                "low_blood_pressure",
                "blood_pressure_valid",
            ],
            errors="ignore",
        ).merge(pressure, on="video_id", how="left", validate="many_to_one")
        systolic = pd.to_numeric(result["high_blood_pressure"], errors="coerce")
        diastolic = pd.to_numeric(result["low_blood_pressure"], errors="coerce")
        valid = pd.to_numeric(result["blood_pressure_valid"], errors="coerce").eq(1)
        if (~valid | systolic.isna() | diastolic.isna()).any():
            raise ValueError("Missing raw systolic/diastolic values for BP regression")
        systolic_threshold = definition["threshold"]["systolic"]
        diastolic_threshold = definition["threshold"]["diastolic"]
        systolic_scale = definition["scale"]["systolic"]
        diastolic_scale = definition["scale"]["diastolic"]
        distance = np.maximum(
            (systolic - systolic_threshold) / systolic_scale,
            (diastolic - diastolic_threshold) / diastolic_scale,
        )
        result["raw_value"] = systolic
        result["systolic_blood_pressure"] = systolic
        result["diastolic_blood_pressure"] = diastolic
        result["score_threshold"] = systolic_threshold
        result["score_scale"] = systolic_scale
    else:
        raw_value = pd.to_numeric(result[definition["value_column"]], errors="coerce")
        if raw_value.isna().any():
            raise ValueError(f"Missing raw values for {target}")
        if target == "hemoglobin_low":
            threshold = np.where(
                result["sex"].astype(str).eq("男"),
                definition["threshold"]["male"],
                definition["threshold"]["other"],
            ).astype(np.float64)
        else:
            threshold = np.full(len(result), definition["threshold"], dtype=np.float64)
        scale = float(definition["scale"])
        if definition["direction"] == "low":
            distance = (threshold - raw_value.to_numpy(np.float64)) / scale
        elif definition["direction"] == "high":
            distance = (raw_value.to_numpy(np.float64) - threshold) / scale
        else:
            raise ValueError(f"Unsupported score direction for {target}")
        result["raw_value"] = raw_value
        result["score_threshold"] = threshold
        result["score_scale"] = scale

    result["standardized_distance"] = np.asarray(distance, dtype=np.float64)
    result["abnormal_score"] = np.arcsinh(result["standardized_distance"])
    if not np.isfinite(result["abnormal_score"]).all():
        raise ValueError(f"Non-finite abnormal scores for {target}")
    return result


def build_task_records(base_manifest, video_summary, target):
    labels = pd.to_numeric(base_manifest[target], errors="coerce")
    invalid_labels = sorted(set(labels.dropna().unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Non-binary values for {target}: {invalid_labels}")
    events = base_manifest.loc[labels.notna()].copy()
    events[target] = labels.loc[events.index].astype(int)
    if events.empty:
        return pd.DataFrame(), [], {
            "target": target, "status": "skipped", "reason": "no non-missing labels"
        }
    audit_rows = _conflicting_video_audit(events, target)
    conflicting_videos = {row["video_id"] for row in audit_rows}
    clean = events.loc[~events["video_id"].astype(str).isin(conflicting_videos)].copy()
    if not clean.groupby("video_id")[target].nunique().le(1).all():
        raise AssertionError(f"Conflicting labels remain for {target}")
    clean["binary_label"] = clean[target].astype(int)
    scored = _score_events(clean, target, video_summary)
    scored["_selection_delta"] = pd.to_numeric(
        scored["match_delta_h"], errors="coerce"
    ).fillna(np.inf)
    records = scored.sort_values(
        ["hospital_id", "video_id", "_selection_delta", "label_time_unix", "sample_id"]
    ).drop_duplicates("video_id", keep="first")
    columns = [
        "hospital_id",
        "video_id",
        "mirror",
        "lab_patient_id",
        "binary_label",
        "raw_value",
        "score_threshold",
        "score_scale",
        "standardized_distance",
        "abnormal_score",
        "source_sample_id",
        "match_delta_h",
        "match_signed_delta_h",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
    ]
    records = records[columns].reset_index(drop=True)
    positive_videos = int(records["binary_label"].eq(1).sum())
    negative_videos = int(records["binary_label"].eq(0).sum())
    positive_distance = records["standardized_distance"].gt(0)
    negative_distance = records["standardized_distance"].lt(0)
    if (positive_distance & records["binary_label"].ne(1)).any():
        raise AssertionError(f"Positive score disagrees with binary label for {target}")
    if (negative_distance & records["binary_label"].ne(0)).any():
        raise AssertionError(f"Negative score disagrees with binary label for {target}")
    summary = {
        "target": target,
        "status": "pending",
        "reason": "",
        "source_events": int(len(events)),
        "source_videos": int(events["video_id"].nunique()),
        "conflicting_videos": int(len(conflicting_videos)),
        "clean_videos": int(len(records)),
        "clean_patients": int(records["hospital_id"].astype(str).nunique()),
        "positive_videos": positive_videos,
        "negative_videos": negative_videos,
        "neutral_boundary_videos": int(records["abnormal_score"].eq(0).sum()),
        "score_min": float(records["abnormal_score"].min()),
        "score_median": float(records["abnormal_score"].median()),
        "score_max": float(records["abnormal_score"].max()),
    }
    if min(positive_videos, negative_videos) < MIN_VIDEOS_PER_CLASS:
        summary.update({
            "status": "skipped",
            "reason": (
                f"fewer than {MIN_VIDEOS_PER_CLASS} clean videos in one class: "
                f"positive={positive_videos}, negative={negative_videos}"
            ),
        })
    return records, audit_rows, summary


def _split_value_columns(target):
    columns = ["raw_value", "abnormal_score"]
    if target == "high_blood_pressure":
        columns.append("diastolic_blood_pressure")
    return columns


def _class_allocation(count):
    if count < 3:
        raise ValueError(f"At least three patients per class are required, got {count}")
    train = max(1, int(round(count * SPLIT_FRACTIONS[0])))
    validation = max(1, int(round(count * SPLIT_FRACTIONS[1])))
    if train + validation > count - 1:
        train = count - validation - 1
    return train, validation, count - train - validation


def _distribution_pair(values, split_codes, first, second):
    first_values = values[split_codes == first]
    second_values = values[split_codes == second]
    global_iqr = max(
        float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        1e-9,
    )
    ks = float(ks_2samp(first_values, second_values).statistic)
    wasserstein = float(wasserstein_distance(first_values, second_values))
    normalized_wasserstein = wasserstein / global_iqr
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
        "wasserstein_iqr": normalized_wasserstein,
        "max_quantile_difference_iqr": quantile_difference,
        "small_sample_rule": bool(small),
        "ks_limit": ks_limit,
        "wasserstein_iqr_limit": wasserstein_limit,
        "passed": bool(
            ks <= ks_limit + 1e-12
            and normalized_wasserstein <= wasserstein_limit + 1e-12
        ),
    }


def _candidate_score(records, patient_assignment, patient_row_indices, value_columns):
    split_codes = patient_assignment[patient_row_indices]
    pair_metrics = []
    for column in value_columns:
        values = pd.to_numeric(records[column], errors="raise").to_numpy(np.float64)
        for first, second in ((0, 1), (0, 2), (1, 2)):
            pair_metrics.append(
                _distribution_pair(values, split_codes, first, second)
            )
    max_ks = max(row["ks"] for row in pair_metrics)
    max_wasserstein = max(row["wasserstein_iqr"] for row in pair_metrics)
    max_quantile = max(
        row["max_quantile_difference_iqr"] for row in pair_metrics
    )
    labels = records["binary_label"].to_numpy(np.float64)
    positive_rates = np.asarray(
        [labels[split_codes == code].mean() for code in range(3)]
    )
    video_fractions = np.asarray(
        [(split_codes == code).mean() for code in range(3)]
    )
    patient_fractions = np.asarray(
        [(patient_assignment == code).mean() for code in range(3)]
    )
    target_fractions = np.asarray(SPLIT_FRACTIONS)
    size_error = float(
        max(
            np.abs(video_fractions - target_fractions).max(),
            np.abs(patient_fractions - target_fractions).max(),
        )
    )
    positive_rate_range = float(positive_rates.max() - positive_rates.min())
    objective = (
        2.0 * max_wasserstein
        + max_ks
        + 0.25 * max_quantile
        + 0.50 * positive_rate_range
        + 0.25 * size_error
    )
    return {
        "objective": float(objective),
        "max_ks": float(max_ks),
        "max_wasserstein_iqr": float(max_wasserstein),
        "max_quantile_difference_iqr": float(max_quantile),
        "positive_rate_range": positive_rate_range,
        "size_fraction_error": size_error,
        "passed": bool(
            all(row["passed"] for row in pair_metrics)
            and size_error <= SPLIT_SIZE_FRACTION_MAX + 1e-12
            and positive_rate_range <= SPLIT_POSITIVE_RATE_RANGE_MAX + 1e-12
        ),
    }


def _distribution_audit(records, target):
    summary_rows, pair_rows = [], []
    value_columns = _split_value_columns(target)
    for column in value_columns:
        values = pd.to_numeric(records[column], errors="raise")
        for split in ("train", "val", "test"):
            selected = values.loc[records["split"].eq(split)].to_numpy(np.float64)
            split_records = records.loc[records["split"].eq(split)]
            summary_rows.append({
                "target": target,
                "variable": column,
                "split": split,
                "videos": int(len(selected)),
                "patients": int(split_records["hospital_id"].nunique()),
                "positive_videos": int(split_records["binary_label"].eq(1).sum()),
                "positive_rate": float(split_records["binary_label"].mean()),
                "mean": float(np.mean(selected)),
                "std": float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0,
                "minimum": float(np.min(selected)),
                "q10": float(np.quantile(selected, 0.10)),
                "q25": float(np.quantile(selected, 0.25)),
                "median": float(np.quantile(selected, 0.50)),
                "q75": float(np.quantile(selected, 0.75)),
                "q90": float(np.quantile(selected, 0.90)),
                "maximum": float(np.max(selected)),
            })
        split_codes = records["split"].map(
            {"train": 0, "val": 1, "test": 2}
        ).to_numpy(np.int8)
        all_values = values.to_numpy(np.float64)
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            row = _distribution_pair(
                all_values,
                split_codes,
                {"train": 0, "val": 1, "test": 2}[first],
                {"train": 0, "val": 1, "test": 2}[second],
            )
            pair_rows.append({
                "target": target,
                "variable": column,
                "split_first": first,
                "split_second": second,
                **row,
            })
    return summary_rows, pair_rows


def add_patient_split(records, target, seed=SEED):
    records = records.copy()
    records["hospital_id"] = records["hospital_id"].astype(str)
    patient_labels = (
        records.groupby("hospital_id", sort=True)["binary_label"].max().reset_index()
    )
    class_counts = patient_labels["binary_label"].value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < MIN_PATIENTS_PER_CLASS:
        return None, (
            f"fewer than {MIN_PATIENTS_PER_CLASS} stratifiable patients in one class: "
            f"{class_counts.to_dict()}"
        ), [], [], {}

    patient_lookup = {
        patient: index
        for index, patient in enumerate(patient_labels["hospital_id"].astype(str))
    }
    patient_row_indices = records["hospital_id"].map(patient_lookup).to_numpy(np.int64)
    labels = patient_labels["binary_label"].to_numpy(np.uint8)
    target_offset = int.from_bytes(
        hashlib.sha256(target.encode("utf-8")).digest()[:4], "little"
    )
    rng = np.random.default_rng((int(seed) + target_offset) % (2**32))
    best_passed = None
    best_overall = None
    value_columns = _split_value_columns(target)
    for candidate_index in range(SPLIT_CANDIDATES):
        assignment = np.empty(len(patient_labels), dtype=np.int8)
        for class_label in (0, 1):
            class_indices = rng.permutation(np.flatnonzero(labels == class_label))
            train_count, validation_count, _ = _class_allocation(len(class_indices))
            assignment[class_indices[:train_count]] = 0
            assignment[
                class_indices[train_count:train_count + validation_count]
            ] = 1
            assignment[
                class_indices[train_count + validation_count:]
            ] = 2
        score = _candidate_score(
            records, assignment, patient_row_indices, value_columns
        )
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
        return None, (
            f"no distribution-balanced split passed after {SPLIT_CANDIDATES} "
            f"candidates; best max_KS={score['max_ks']:.4f}, "
            f"best max_Wasserstein/IQR={score['max_wasserstein_iqr']:.4f}"
        ), [], [], {
            "target": target,
            "accepted": False,
            "candidates": SPLIT_CANDIDATES,
            **score,
        }

    _, assignment, selected_score, candidate_index = best_passed
    result = records.copy()
    result["split"] = np.asarray(("train", "val", "test"))[
        assignment[patient_row_indices]
    ]
    if result["split"].isna().any():
        raise AssertionError(f"Unassigned patients for {target}")
    patient_sets = {
        split: set(result.loc[result["split"].eq(split), "hospital_id"])
        for split in ("train", "val", "test")
    }
    if any(patient_sets[a] & patient_sets[b] for a, b in (
        ("train", "val"), ("train", "test"), ("val", "test")
    )):
        raise AssertionError(f"Patient leakage detected for {target}")
    if len(result) != len(records) or result["video_id"].nunique() != len(records):
        raise AssertionError(f"Video loss or duplication detected for {target}")
    summary_rows, pair_rows = _distribution_audit(result, target)
    if not all(row["passed"] for row in pair_rows):
        raise AssertionError(f"Selected split failed distribution audit for {target}")
    selection = {
        "target": target,
        "accepted": True,
        "candidates": SPLIT_CANDIDATES,
        "selected_candidate_index": int(candidate_index),
        "value_columns": value_columns,
        **selected_score,
    }
    return result.reset_index(drop=True), "", summary_rows, pair_rows, selection


def _plot_split_distributions(task_records, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_colors = {"train": "#4C78A8", "val": "#F2A541", "test": "#59A14F"}
    figure, axes = plt.subplots(
        len(task_records),
        2,
        figsize=(13, 3.3 * len(task_records)),
        squeeze=False,
    )
    for row, (target, records) in enumerate(task_records.items()):
        for column, variable in enumerate(("raw_value", "abnormal_score")):
            axis = axes[row, column]
            for split in ("train", "val", "test"):
                values = np.sort(
                    pd.to_numeric(
                        records.loc[records["split"].eq(split), variable],
                        errors="raise",
                    ).to_numpy(np.float64)
                )
                ecdf = np.arange(1, len(values) + 1) / len(values)
                axis.step(
                    values,
                    ecdf,
                    where="post",
                    color=split_colors[split],
                    linewidth=1.4,
                    label=split,
                )
            axis.set_title(f"{target} | {variable}")
            axis.set_xlabel(variable)
            axis.set_ylabel("Empirical CDF")
            axis.grid(alpha=0.22)
            axis.legend()
    figure.suptitle("Patient-disjoint split distribution audit", fontsize=14)
    figure.tight_layout()
    figure.savefig(
        os.path.join(output_dir, "split_distribution.png"),
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def prepare_tasks(base_manifest, video_summary, output_dir, targets=TARGETS, seed=SEED):
    os.makedirs(os.path.join(output_dir, "task_records"), exist_ok=True)
    summaries, conflicts, task_records = [], [], {}
    distribution_summaries, distribution_pairs, split_selections = [], [], []
    for target in targets:
        records, audit_rows, summary = build_task_records(
            base_manifest, video_summary, target
        )
        conflicts.extend(audit_rows)
        if summary["status"] != "skipped":
            records, reason, audit_rows, pair_rows, selection = add_patient_split(
                records, target, seed
            )
            distribution_summaries.extend(audit_rows)
            distribution_pairs.extend(pair_rows)
            split_selections.append(selection)
            if records is None:
                summary.update({"status": "skipped", "reason": reason})
            else:
                summary["status"] = "ready"
                for split in ("train", "val", "test"):
                    summary[f"{split}_patients"] = int(
                        records.loc[records["split"].eq(split), "hospital_id"].nunique()
                    )
                    summary[f"{split}_videos"] = int(records["split"].eq(split).sum())
                records.to_csv(
                    os.path.join(output_dir, "task_records", f"{target}.csv"), index=False
                )
                task_records[target] = records
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    conflict_frame = pd.DataFrame(conflicts)
    summary_frame.to_csv(os.path.join(output_dir, "task_summary.csv"), index=False)
    conflict_frame.to_csv(os.path.join(output_dir, "conflicting_videos.csv"), index=False)
    pd.DataFrame(distribution_summaries).to_csv(
        os.path.join(output_dir, "split_distribution_audit.csv"), index=False
    )
    pd.DataFrame(distribution_pairs).to_csv(
        os.path.join(output_dir, "split_distribution_pairwise.csv"), index=False
    )
    with open(
        os.path.join(output_dir, "split_assignment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "algorithm": (
                    "patient-group class-stratified candidate search with "
                    "video-level continuous-distribution selection"
                ),
                "seed": int(seed),
                "candidate_count": SPLIT_CANDIDATES,
                "target_fractions": {
                    "train": SPLIT_FRACTIONS[0],
                    "val": SPLIT_FRACTIONS[1],
                    "test": SPLIT_FRACTIONS[2],
                },
                "acceptance_thresholds": {
                    "size_fraction_error_max": SPLIT_SIZE_FRACTION_MAX,
                    "positive_rate_range_max": SPLIT_POSITIVE_RATE_RANGE_MAX,
                    "regular": {
                        "minimum_pair_n": SPLIT_SMALL_N,
                        "ks_max": SPLIT_KS_MAX,
                        "wasserstein_iqr_max": SPLIT_WASSERSTEIN_IQR_MAX,
                    },
                    "small_sample": {
                        "ks_max": SPLIT_SMALL_KS_MAX,
                        "wasserstein_iqr_max": SPLIT_SMALL_WASSERSTEIN_IQR_MAX,
                    },
                },
                "selection_objective": (
                    "2*max_wasserstein_iqr + max_ks + "
                    "0.25*max_quantile_difference_iqr + "
                    "0.50*positive_rate_range + 0.25*size_fraction_error"
                ),
                "target_results": split_selections,
            },
            handle,
            indent=2,
        )
    if task_records:
        _plot_split_distributions(task_records, output_dir)
    return task_records, summary_frame, conflict_frame


class GroupedFrameViewSampler(Sampler):
    """Shuffle contiguous frame chunks while keeping each frame's views together."""

    def __init__(self, dataset, chunk_size=FRAME_SHUFFLE_CHUNK_SIZE):
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

    def __iter__(self):
        frame_count = self.dataset.frame_count
        chunk_count = (frame_count + self.chunk_size - 1) // self.chunk_size
        for chunk_index in torch.randperm(chunk_count).tolist():
            start = chunk_index * self.chunk_size
            end = min(start + self.chunk_size, frame_count)
            for frame_index in range(start, end):
                if self.dataset.expand_all_views:
                    yield frame_index
                else:
                    sample_offset = frame_index * len(self.dataset.views)
                    for view_index in torch.randperm(len(self.dataset.views)).tolist():
                        yield sample_offset + view_index

    def __len__(self):
        return len(self.dataset)


class AllFramesDataset(Dataset):
    def __init__(
        self,
        frame_index,
        video_records,
        views=("original",),
        interpolation="bilinear",
        expand_all_views=False,
    ):
        self.index = frame_index
        self.video_records = video_records.reset_index(drop=True).copy()
        self.views = tuple(views)
        self.interpolation = interpolation
        self.expand_all_views = bool(expand_all_views)
        if self.expand_all_views and len(self.views) < 2:
            raise ValueError("expand_all_views requires multiple training views")
        frame_indices, frame_video_rows = [], []
        record_index_videos = []
        for record_index, row in enumerate(self.video_records.itertuples(index=False)):
            index_video = self.index.video_lookup[str(row.video_id)]
            start, end = self.index.frame_range(row.video_id)
            frame_indices.append(np.arange(start, end, dtype=np.int64))
            frame_video_rows.append(np.full(end - start, record_index, dtype=np.int32))
            record_index_videos.append(index_video)
        self.frame_indices = np.concatenate(frame_indices)
        self.frame_video_rows = np.concatenate(frame_video_rows)
        self.record_index_videos = np.asarray(record_index_videos, dtype=np.int32)
        self.labels_by_video = self.video_records["binary_label"].to_numpy(np.float32)
        self.frame_count = len(self.frame_indices)
        self._handles = OrderedDict()
        self._decoded_cache = OrderedDict()

    def __len__(self):
        if self.expand_all_views:
            return self.frame_count
        return self.frame_count * len(self.views)

    @property
    def model_input_count(self):
        return self.frame_count * len(self.views)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_decoded_cache"] = OrderedDict()
        return state

    def _handle(self, path):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > MAX_OPEN_FILES_PER_WORKER:
            _, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def _decode(self, global_frame_index, video_row):
        cached = self._decoded_cache.pop(global_frame_index, None)
        if cached is not None:
            self._decoded_cache[global_frame_index] = cached
            return cached
        index_video = self.record_index_videos[video_row]
        path = str(self.index.video_paths[index_video])
        start = int(self.index.starts[global_frame_index])
        end = int(self.index.ends[global_frame_index])
        handle = self._handle(path)
        handle.seek(start)
        payload = handle.read(end - start)
        try:
            encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            tensor = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
            expected_shape = (3, SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"unexpected decoded shape {tuple(tensor.shape)}")
        except Exception as exc:
            raise RuntimeError(
                f"Decode failed path={path} frame={global_frame_index} "
                f"offsets={start}:{end}: {exc}"
            ) from exc
        self._decoded_cache[global_frame_index] = tensor
        while len(self._decoded_cache) > DECODE_CACHE_FRAMES:
            self._decoded_cache.popitem(last=False)
        return tensor

    def __getitem__(self, index):
        if self.expand_all_views:
            base_frame_index = index
            view_index = torch.arange(len(self.views), dtype=torch.uint8)
        else:
            base_frame_index = index // len(self.views)
            view_index = torch.tensor(index % len(self.views), dtype=torch.uint8)
        global_frame_index = int(self.frame_indices[base_frame_index])
        video_row = int(self.frame_video_rows[base_frame_index])
        image = self._decode(global_frame_index, video_row)
        label = self.labels_by_video[video_row]
        return (
            image,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(base_frame_index, dtype=torch.long),
            view_index,
        )

    def frame_labels(self):
        return self.labels_by_video[self.frame_video_rows]

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._decoded_cache.clear()

    def __del__(self):
        self.close()
