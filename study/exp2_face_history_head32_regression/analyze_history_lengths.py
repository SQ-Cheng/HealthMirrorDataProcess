"""Describe prior-lab sequence lengths and characteristics of long-history patients."""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.exp2_lab_multimodal.build_dataset import (
    _normalize_hospital_id,
    _parse_datetime_to_unix,
)
from study.exp2_lab_multimodal.config import LAB_CSV

from .config import OUTPUT_DIR


HISTORY_DIR = os.path.join(OUTPUT_DIR, "history_records")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "history_length_analysis")
TABLE_DIR = os.path.join(ANALYSIS_DIR, "tables")
FIGURE_DIR = os.path.join(ANALYSIS_DIR, "figures")
TARGETS = ("hemoglobin_low", "po2_low")
TARGET_LABELS = {"hemoglobin_low": "Hemoglobin", "po2_low": "PO2"}
LENGTH_BINS = (-0.5, 0.5, 9.5, 29.5, 49.5, 99.5, np.inf)
LENGTH_LABELS = ("0", "1-9", "10-29", "30-49", "50-99", "100+")


def _first_nonempty(values):
    values = values.astype(str).str.strip()
    values = values[~values.isin(("", "nan", "None"))]
    return values.iloc[0] if len(values) else ""


def _age_number(value):
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def _episode_metadata():
    columns = [
        "首页病案号",
        "首页性别",
        "首页就诊时年龄",
        "首页入院时间",
        "首页出院时间",
        "首页住院天数",
        "首页入院科室",
        "首页手术操作名称",
        "手术开始日期",
        "手术结束日期",
    ]
    raw = pd.read_csv(LAB_CSV, dtype=str, keep_default_na=False, usecols=columns)
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["episode_admission_time_unix"] = _parse_datetime_to_unix(raw["首页入院时间"])
    raw["episode_discharge_time_unix"] = _parse_datetime_to_unix(raw["首页出院时间"])
    raw["surgery_start_unix"] = _parse_datetime_to_unix(raw["手术开始日期"])
    raw["surgery_end_unix"] = _parse_datetime_to_unix(raw["手术结束日期"])
    valid = (
        raw["hospital_id"].ne("")
        & raw["episode_admission_time_unix"].notna()
        & raw["episode_discharge_time_unix"].notna()
        & raw["episode_discharge_time_unix"].ge(raw["episode_admission_time_unix"])
    )
    raw = raw.loc[valid].copy()
    group_columns = [
        "hospital_id",
        "episode_admission_time_unix",
        "episode_discharge_time_unix",
    ]
    rows = []
    for keys, group in raw.groupby(group_columns, sort=True):
        admission, discharge = float(keys[1]), float(keys[2])
        surgery_start = pd.to_numeric(group["surgery_start_unix"], errors="coerce")
        valid_surgery = surgery_start.between(admission, discharge, inclusive="both")
        rows.append(
            {
                "hospital_id": str(keys[0]),
                "episode_admission_time_unix": admission,
                "episode_discharge_time_unix": discharge,
                "sex": _first_nonempty(group["首页性别"]),
                "age_years": _age_number(_first_nonempty(group["首页就诊时年龄"])),
                "admission_department": _first_nonempty(group["首页入院科室"]),
                "length_of_stay_days": (discharge - admission) / 86400.0,
                "has_valid_surgery": bool(valid_surgery.any()),
                "surgery_name": _first_nonempty(group["首页手术操作名称"]),
            }
        )
    return pd.DataFrame(rows)


def _enriched_samples(target, episodes):
    summary = pd.read_csv(
        os.path.join(HISTORY_DIR, f"{target}_summary.csv"),
        dtype={"hospital_id": str, "video_id": str},
    )
    history = pd.read_csv(
        os.path.join(HISTORY_DIR, f"{target}.csv"),
        dtype={"hospital_id": str, "video_id": str},
    )
    if len(history):
        event_summary = history.groupby("video_id", as_index=False).agg(
            unique_report_times=("history_lab_time_unix", "nunique"),
            distinct_item_names=("history_item_name", "nunique"),
            history_span_hours=(
                "history_minus_current_hours",
                lambda values: float(-values.min()),
            ),
        )
        item_pairs = history.groupby("video_id").apply(
            lambda group: int(
                group.duplicated(["history_lab_time_unix"], keep=False).sum()
            ),
            include_groups=False,
        ).rename("rows_at_repeated_report_times").reset_index()
        event_summary = event_summary.merge(item_pairs, on="video_id", validate="one_to_one")
    else:
        event_summary = pd.DataFrame(columns=["video_id"])
    result = summary.merge(event_summary, on="video_id", how="left", validate="one_to_one")
    fill_zero = [
        "unique_report_times",
        "distinct_item_names",
        "history_span_hours",
        "rows_at_repeated_report_times",
    ]
    result[fill_zero] = result[fill_zero].fillna(0)
    result = result.merge(
        episodes,
        on="hospital_id",
        how="left",
        validate="many_to_many",
    )
    inside = (
        result["current_lab_time_unix"].ge(result["episode_admission_time_unix"])
        & result["current_lab_time_unix"].le(result["episode_discharge_time_unix"])
    )
    result = result.loc[inside].copy()
    if result["video_id"].duplicated().any():
        raise ValueError(f"Ambiguous episode metadata for {target}")
    result["target"] = target
    result["current_day_from_admission"] = (
        result["current_lab_time_unix"] - result["episode_admission_time_unix"]
    ) / 86400.0
    result["model_rows_per_unique_time"] = result["history_count"].div(
        result["unique_report_times"].replace(0, np.nan)
    )
    result["unique_times_per_elapsed_day"] = result["unique_report_times"].div(
        result["current_day_from_admission"].clip(lower=0.25)
    )
    result["sequence_bin"] = pd.cut(
        result["history_count"],
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
    )
    return result


def _quantile_rows(samples):
    rows = []
    for target in TARGETS:
        selected = samples.loc[samples["target"].eq(target)]
        for split in ("all", "train", "val", "test"):
            group = selected if split == "all" else selected.loc[selected["split"].eq(split)]
            for variable in ("history_count", "unique_report_times"):
                values = group[variable].to_numpy(np.float64)
                quantiles = np.quantile(values, (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1))
                rows.append(
                    {
                        "target": target,
                        "split": split,
                        "variable": variable,
                        "n": len(values),
                        **{
                            name: float(value)
                            for name, value in zip(
                                ("min", "q25", "median", "q75", "q90", "q95", "q99", "max"),
                                quantiles,
                            )
                        },
                    }
                )
    return pd.DataFrame(rows)


def _patient_max_table(samples):
    ordered = samples.sort_values(
        ["target", "hospital_id", "history_count", "current_lab_time_unix"],
        ascending=[True, True, False, False],
    )
    result = ordered.drop_duplicates(["target", "hospital_id"], keep="first").copy()
    videos_per_patient = samples.groupby(["target", "hospital_id"]).size().rename(
        "eligible_videos_for_patient"
    )
    result = result.merge(
        videos_per_patient.reset_index(),
        on=["target", "hospital_id"],
        validate="one_to_one",
    )
    return result


def _characteristics(patient_max):
    rows = []
    cohorts = (
        ("all", lambda frame: np.ones(len(frame), dtype=bool)),
        ("<50", lambda frame: frame["history_count"].lt(50)),
        (">=50", lambda frame: frame["history_count"].ge(50)),
        (">=100", lambda frame: frame["history_count"].ge(100)),
    )
    for target in TARGETS:
        selected = patient_max.loc[patient_max["target"].eq(target)]
        for cohort, selector in cohorts:
            group = selected.loc[selector(selected)]
            rows.append(
                {
                    "target": target,
                    "patient_cohort": cohort,
                    "patients": len(group),
                    "female_fraction": float(group["sex"].eq("女").mean()) if len(group) else np.nan,
                    "median_age_years": float(group["age_years"].median()) if len(group) else np.nan,
                    "median_length_of_stay_days": float(group["length_of_stay_days"].median()) if len(group) else np.nan,
                    "median_current_day_from_admission": float(group["current_day_from_admission"].median()) if len(group) else np.nan,
                    "valid_surgery_fraction": float(group["has_valid_surgery"].mean()) if len(group) else np.nan,
                    "median_history_rows": float(group["history_count"].median()) if len(group) else np.nan,
                    "median_unique_report_times": float(group["unique_report_times"].median()) if len(group) else np.nan,
                    "median_rows_per_unique_time": float(group["model_rows_per_unique_time"].median()) if len(group) else np.nan,
                    "median_unique_times_per_elapsed_day": float(group["unique_times_per_elapsed_day"].median()) if len(group) else np.nan,
                    "median_eligible_videos_per_patient": float(group["eligible_videos_for_patient"].median()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _department_table(patient_max):
    rows = []
    for target in TARGETS:
        selected = patient_max.loc[patient_max["target"].eq(target)]
        for cohort, group in (
            ("all", selected),
            (">=50", selected.loc[selected["history_count"].ge(50)]),
            (">=100", selected.loc[selected["history_count"].ge(100)]),
        ):
            counts = group["admission_department"].replace("", "Unknown").value_counts()
            for department, count in counts.items():
                rows.append(
                    {
                        "target": target,
                        "patient_cohort": cohort,
                        "admission_department": department,
                        "patients": int(count),
                        "fraction_within_cohort": float(count / len(group)),
                    }
                )
    return pd.DataFrame(rows)


def _plot(samples, patient_max):
    colors = {"hemoglobin_low": "#2878B5", "po2_low": "#D95F02"}
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for target in TARGETS:
        selected = samples.loc[samples["target"].eq(target)]
        axes[0, 0].hist(
            selected["history_count"],
            bins=np.arange(0, 126, 5),
            alpha=0.55,
            color=colors[target],
            label=TARGET_LABELS[target],
        )
        values = np.sort(selected["history_count"].to_numpy())
        axes[0, 1].step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            color=colors[target],
            label=TARGET_LABELS[target],
        )
        p = patient_max.loc[patient_max["target"].eq(target)]
        axes[1, 0].scatter(
            p["current_day_from_admission"],
            p["history_count"],
            s=18,
            alpha=0.5,
            color=colors[target],
            label=TARGET_LABELS[target],
        )
        axes[1, 1].scatter(
            p["unique_report_times"],
            p["history_count"],
            s=18,
            alpha=0.5,
            color=colors[target],
            label=TARGET_LABELS[target],
        )
    axes[0, 0].set(xlabel="History rows supplied to model", ylabel="Video samples", title="Sequence-length distribution")
    axes[0, 1].set(xlabel="History rows supplied to model", ylabel="Empirical CDF", title="Sequence-length ECDF")
    axes[1, 0].set(xlabel="Current label day from admission", ylabel="Maximum history rows per patient", title="Later labels accumulate more history")
    axes[1, 1].set(xlabel="Unique prior report times", ylabel="Maximum history rows per patient", title="Rows versus unique report times")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Prior-lab history length audit", fontsize=15)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURE_DIR, "history_length_audit.png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    episodes = _episode_metadata()
    samples = pd.concat(
        [_enriched_samples(target, episodes) for target in TARGETS],
        ignore_index=True,
    )
    distribution = (
        samples.groupby(["target", "sequence_bin"], observed=False)
        .size()
        .rename("video_samples")
        .reset_index()
    )
    distribution["fraction_within_target"] = distribution["video_samples"].div(
        distribution.groupby("target")["video_samples"].transform("sum")
    )
    patient_max = _patient_max_table(samples)
    samples.to_csv(os.path.join(TABLE_DIR, "sample_history_characteristics.csv"), index=False)
    patient_max.to_csv(os.path.join(TABLE_DIR, "patient_max_history.csv"), index=False)
    distribution.to_csv(os.path.join(TABLE_DIR, "sequence_length_distribution.csv"), index=False)
    _quantile_rows(samples).to_csv(os.path.join(TABLE_DIR, "sequence_length_quantiles.csv"), index=False)
    _characteristics(patient_max).to_csv(os.path.join(TABLE_DIR, "long_history_characteristics.csv"), index=False)
    _department_table(patient_max).to_csv(os.path.join(TABLE_DIR, "long_history_departments.csv"), index=False)
    _plot(samples, patient_max)
    print(f"Saved history-length audit to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
