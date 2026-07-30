"""Global patient split, missing-label masks, and all-frame streaming dataset."""

import json
import os

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.data import (
    AllFramesDataset,
    GroupedFrameViewSampler,
    _distribution_audit,
    _distribution_pair,
    _plot_split_distributions,
    build_task_records,
    validate_source_data,
)

from .config import (
    SEED,
    SPLIT_CANDIDATES,
    SPLIT_FRACTIONS,
    SPLIT_POSITIVE_RATE_RANGE_MAX,
    SPLIT_SIZE_FRACTION_MAX,
    TARGETS,
)


IDENTITY_COLUMNS = ("hospital_id", "video_id", "mirror", "lab_patient_id")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def _task_value_columns(target):
    columns = ["raw_value", "abnormal_score"]
    if target == "high_blood_pressure":
        columns.append("diastolic_blood_pressure")
    return columns


def _merge_task_records(task_records):
    identity = pd.concat(
        [records[list(IDENTITY_COLUMNS)] for records in task_records.values()],
        ignore_index=True,
    )
    for column in IDENTITY_COLUMNS[1:]:
        inconsistent = identity.groupby("video_id")[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            videos = inconsistent.index[inconsistent].astype(str).tolist()
            raise ValueError(f"Inconsistent {column} for videos: {videos[:5]}")
    wide = identity.drop_duplicates("video_id").reset_index(drop=True)
    for target, records in task_records.items():
        task_columns = [column for column in records if column not in IDENTITY_COLUMNS]
        renamed = records[list(IDENTITY_COLUMNS) + task_columns].rename(
            columns={column: f"{target}__{column}" for column in task_columns}
        )
        wide = wide.merge(
            renamed,
            on=list(IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        wide[f"{target}__mask"] = wide[f"{target}__abnormal_score"].notna().astype(
            np.uint8
        )
    mask_columns = [f"{target}__mask" for target in TARGETS]
    if wide[mask_columns].sum(axis=1).lt(1).any():
        raise AssertionError("A union video has no usable target")
    if wide["video_id"].duplicated().any():
        raise AssertionError("Duplicate videos remain after the task outer join")
    return wide


def _candidate_metrics(task_records, patients, assignment):
    patient_lookup = {patient: index for index, patient in enumerate(patients)}
    pair_rows = []
    task_rows = []
    class_coverage = True
    for target, records in task_records.items():
        row_indices = records["hospital_id"].astype(str).map(patient_lookup).to_numpy(
            np.int64
        )
        split_codes = assignment[row_indices]
        labels = records["binary_label"].to_numpy(np.uint8)
        split_counts = np.asarray(
            [(split_codes == code).sum() for code in range(3)], dtype=np.int64
        )
        positive_counts = np.asarray(
            [labels[split_codes == code].sum() for code in range(3)], dtype=np.int64
        )
        negative_counts = split_counts - positive_counts
        class_coverage &= bool((positive_counts > 0).all() and (negative_counts > 0).all())
        positive_rates = positive_counts / split_counts
        video_fractions = split_counts / split_counts.sum()
        task_rows.append(
            {
                "target": target,
                "positive_rate_range": float(
                    positive_rates.max() - positive_rates.min()
                ),
                "size_fraction_error": float(
                    np.abs(video_fractions - np.asarray(SPLIT_FRACTIONS)).max()
                ),
            }
        )
        for column in _task_value_columns(target):
            values = pd.to_numeric(records[column], errors="raise").to_numpy(np.float64)
            for first, second in ((0, 1), (0, 2), (1, 2)):
                pair_rows.append(
                    _distribution_pair(values, split_codes, first, second)
                )

    max_ks = max(row["ks"] for row in pair_rows)
    max_wasserstein = max(row["wasserstein_iqr"] for row in pair_rows)
    max_quantile = max(
        row["max_quantile_difference_iqr"] for row in pair_rows
    )
    positive_rate_range = max(row["positive_rate_range"] for row in task_rows)
    task_size_error = max(row["size_fraction_error"] for row in task_rows)
    patient_fractions = np.asarray(
        [(assignment == code).mean() for code in range(3)]
    )
    size_error = float(
        max(
            task_size_error,
            np.abs(patient_fractions - np.asarray(SPLIT_FRACTIONS)).max(),
        )
    )
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
        "positive_rate_range": float(positive_rate_range),
        "size_fraction_error": size_error,
        "all_task_split_classes_present": bool(class_coverage),
        "passed": bool(
            class_coverage
            and all(row["passed"] for row in pair_rows)
            and size_error <= SPLIT_SIZE_FRACTION_MAX + 1e-12
            and positive_rate_range <= SPLIT_POSITIVE_RATE_RANGE_MAX + 1e-12
        ),
    }


def add_global_patient_split(task_records, wide_records, seed=SEED):
    patients = np.asarray(
        sorted(wide_records["hospital_id"].astype(str).unique()), dtype=str
    )
    patient_count = len(patients)
    train_count = int(round(patient_count * SPLIT_FRACTIONS[0]))
    val_count = int(round(patient_count * SPLIT_FRACTIONS[1]))
    if min(train_count, val_count, patient_count - train_count - val_count) < 1:
        raise ValueError(f"Too few patients for three splits: {patient_count}")

    rng = np.random.default_rng(seed)
    best_passed = None
    best_overall = None
    for candidate_index in range(SPLIT_CANDIDATES):
        permutation = rng.permutation(patient_count)
        assignment = np.full(patient_count, 2, dtype=np.int8)
        assignment[permutation[:train_count]] = 0
        assignment[permutation[train_count:train_count + val_count]] = 1
        score = _candidate_metrics(task_records, patients, assignment)
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
            f"No valid global split among {SPLIT_CANDIDATES} candidates; "
            f"best max_KS={score['max_ks']:.4f}, "
            f"max_Wasserstein/IQR={score['max_wasserstein_iqr']:.4f}, "
            f"class_coverage={score['all_task_split_classes_present']}"
        )

    _, assignment, score, candidate_index = best_passed
    patient_split = pd.DataFrame(
        {
            "hospital_id": patients,
            "split": np.asarray(SPLIT_NAMES)[assignment],
        }
    )
    patient_to_split = dict(
        zip(patient_split["hospital_id"], patient_split["split"])
    )
    wide = wide_records.copy()
    wide["split"] = wide["hospital_id"].astype(str).map(patient_to_split)
    split_tasks = {}
    summary_rows, pair_rows = [], []
    for target, records in task_records.items():
        selected = records.copy()
        selected["split"] = selected["hospital_id"].astype(str).map(patient_to_split)
        if selected["split"].isna().any():
            raise AssertionError(f"Unassigned patient in {target}")
        task_summary, task_pairs = _distribution_audit(selected, target)
        if not all(row["passed"] for row in task_pairs):
            raise AssertionError(f"Selected global split failed audit for {target}")
        summary_rows.extend(task_summary)
        pair_rows.extend(task_pairs)
        split_tasks[target] = selected.reset_index(drop=True)

    patient_sets = {
        split: set(wide.loc[wide["split"].eq(split), "hospital_id"])
        for split in SPLIT_NAMES
    }
    if any(
        patient_sets[first] & patient_sets[second]
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("Patient leakage detected in global split")
    if len(wide) != len(wide_records) or wide["video_id"].nunique() != len(wide):
        raise AssertionError("Video loss or duplication detected after global split")

    selection = {
        "schema_version": 1,
        "algorithm": (
            "global patient-group candidate search over the union of all task videos"
        ),
        "seed": int(seed),
        "candidate_count": SPLIT_CANDIDATES,
        "selected_candidate_index": int(candidate_index),
        "target_fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS)),
        "acceptance": {
            "every_task_has_both_binary_signs_in_every_split": True,
            "all_distribution_pairs_pass": True,
            "size_fraction_error_max": SPLIT_SIZE_FRACTION_MAX,
            "positive_rate_range_max": SPLIT_POSITIVE_RATE_RANGE_MAX,
        },
        "selection_objective": (
            "2*max_wasserstein_iqr + max_ks + "
            "0.25*max_quantile_difference_iqr + "
            "0.50*positive_rate_range + 0.25*size_fraction_error"
        ),
        **score,
    }
    return (
        wide.reset_index(drop=True),
        split_tasks,
        patient_split,
        pd.DataFrame(summary_rows),
        pd.DataFrame(pair_rows),
        selection,
    )


def prepare_multitask_data(
    base_manifest, video_summary, output_dir, targets=TARGETS, seed=SEED
):
    if tuple(targets) != tuple(TARGETS):
        raise ValueError("The multi-output head requires the configured five-task order")
    os.makedirs(os.path.join(output_dir, "task_records"), exist_ok=True)
    task_records, conflicts, summaries = {}, [], []
    for target in targets:
        records, audit_rows, summary = build_task_records(
            base_manifest, video_summary, target
        )
        if summary["status"] == "skipped":
            raise RuntimeError(f"Task is not trainable: {target}: {summary['reason']}")
        task_records[target] = records
        conflicts.extend(audit_rows)
        summaries.append(summary)

    wide = _merge_task_records(task_records)
    (
        wide,
        task_records,
        patient_split,
        distribution_summary,
        distribution_pairs,
        selection,
    ) = add_global_patient_split(task_records, wide, seed)

    for summary in summaries:
        target = summary["target"]
        records = task_records[target]
        summary["status"] = "ready"
        for split in SPLIT_NAMES:
            subset = records.loc[records["split"].eq(split)]
            summary[f"{split}_patients"] = int(subset["hospital_id"].nunique())
            summary[f"{split}_videos"] = int(len(subset))
        records.to_csv(
            os.path.join(output_dir, "task_records", f"{target}.csv"), index=False
        )

    wide.to_csv(os.path.join(output_dir, "multitask_records.csv"), index=False)
    patient_split.to_csv(os.path.join(output_dir, "patient_split.csv"), index=False)
    pd.DataFrame(summaries).to_csv(
        os.path.join(output_dir, "task_summary.csv"), index=False
    )
    pd.DataFrame(
        conflicts,
        columns=(
            "target",
            "video_id",
            "hospital_id",
            "positive_event_count",
            "negative_event_count",
            "event_count",
            "action",
        ),
    ).to_csv(os.path.join(output_dir, "conflicting_videos.csv"), index=False)
    distribution_summary.to_csv(
        os.path.join(output_dir, "split_distribution_audit.csv"), index=False
    )
    distribution_pairs.to_csv(
        os.path.join(output_dir, "split_distribution_pairwise.csv"), index=False
    )
    with open(
        os.path.join(output_dir, "split_assignment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(selection, handle, indent=2)
    _plot_split_distributions(task_records, output_dir)
    return wide, task_records, pd.DataFrame(summaries), selection


class MultiTaskAllFramesDataset(AllFramesDataset):
    """Decode each frame once and return all observed targets plus their mask."""

    def __init__(self, frame_index, video_records, views=("original",), **kwargs):
        records = video_records.reset_index(drop=True).copy()
        parent_records = records.copy()
        parent_records["abnormal_score"] = 0.0
        super().__init__(frame_index, parent_records, views=views, **kwargs)
        self.video_records = records
        target_columns = [f"{target}__abnormal_score" for target in TARGETS]
        mask_columns = [f"{target}__mask" for target in TARGETS]
        self.targets_by_video = (
            records[target_columns].fillna(0.0).to_numpy(np.float32)
        )
        self.masks_by_video = records[mask_columns].to_numpy(np.float32)
        if not np.isfinite(self.targets_by_video).all():
            raise ValueError("Non-finite observed regression targets")
        if not np.isin(self.masks_by_video, (0.0, 1.0)).all():
            raise ValueError("Invalid target mask")
        if (self.masks_by_video.sum(axis=1) < 1).any():
            raise ValueError("A dataset video has no observed target")

    def __getitem__(self, index):
        image, _, frame_row, view_index = super().__getitem__(index)
        video_row = int(self.frame_video_rows[int(frame_row)])
        return (
            image,
            torch.tensor(self.targets_by_video[video_row], dtype=torch.float32),
            torch.tensor(self.masks_by_video[video_row], dtype=torch.float32),
            frame_row,
            view_index,
        )

    def frame_targets_and_masks(self):
        return (
            self.targets_by_video[self.frame_video_rows],
            self.masks_by_video[self.frame_video_rows],
        )


__all__ = [
    "GroupedFrameViewSampler",
    "MultiTaskAllFramesDataset",
    "prepare_multitask_data",
    "validate_source_data",
]
