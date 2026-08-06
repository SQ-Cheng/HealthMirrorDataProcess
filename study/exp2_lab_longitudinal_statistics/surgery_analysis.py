"""Surgery-aligned laboratory trajectory analysis."""

from collections import Counter
import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SURGERY_COLUMNS = (
    "手术开始日期",
    "手术结束日期",
    "首页手术操作名称",
)

PHASES = (
    ("pre_gt7d", "Pre >7d"),
    ("pre_3_7d", "Pre 3-7d"),
    ("pre_1_3d", "Pre 1-3d"),
    ("pre_lt24h", "Pre <24h"),
    ("intraop", "Intra-op"),
    ("post_0_6h", "Post 0-6h"),
    ("post_6_24h", "Post 6-24h"),
    ("post_1_2d", "Post 1-2d"),
    ("post_2_3d", "Post 2-3d"),
    ("post_3_7d", "Post 3-7d"),
    ("post_gt7d", "Post >7d"),
)
PHASE_ORDER = {phase: index for index, (phase, _) in enumerate(PHASES)}
PHASE_LABELS = dict(PHASES)

ENDPOINTS = (
    "last_preop_7d",
    "intraop",
    "post_0_24h",
    "post_3_7d",
    "predischarge",
)
CONTRASTS = (
    ("preop_to_intraop", "last_preop_7d", "intraop"),
    ("preop_to_post_0_24h", "last_preop_7d", "post_0_24h"),
    ("post_0_24h_to_post_3_7d", "post_0_24h", "post_3_7d"),
    ("preop_to_predischarge", "last_preop_7d", "predischarge"),
)
CONTRAST_LABELS = {
    "preop_to_intraop": "Pre-op -> intra-op",
    "preop_to_post_0_24h": "Pre-op -> post 0-24h",
    "post_0_24h_to_post_3_7d": "Post 0-24h -> post 3-7d",
    "preop_to_predischarge": "Pre-op -> pre-discharge",
}

PROCEDURE_ENGLISH = {
    "冠状动脉旁路移植术": "CABG",
    "单根导管冠状动脉造影": "Single-catheter coronary angiography",
    "经皮冠状动脉球囊扩张成形术": "Coronary balloon angioplasty",
    "脑动脉造影": "Cerebral angiography",
    "锁骨下动脉造影": "Subclavian angiography",
    "经皮冠状动脉药物洗脱支架置入术": "Drug-eluting coronary stent",
    "冠状动脉造影术": "Coronary angiography",
    "直视下冠状动脉内膜剥脱术": "Open coronary endarterectomy",
    "主动脉弓造影": "Aortic arch angiography",
    "经皮颈动脉支架置入术": "Carotid artery stenting",
    "两根导管冠状动脉造影": "Two-catheter coronary angiography",
    "多根导管冠状动脉造影": "Multi-catheter coronary angiography",
}


def _clean_token(value):
    text = str(value).strip()
    return "" if text in {"", "-", "nan", "NaT", "None"} else text


def _split_tokens(value):
    return [_clean_token(token) for token in str(value).split("^")]


def _benjamini_hochberg(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _rank_biserial(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def _safe_wilcoxon(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or np.allclose(values, 0.0):
        return 0.0, 1.0
    try:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def _bootstrap_median_ci(values, rng, replicates):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(
        values,
        size=(replicates, len(values)),
        replace=True,
    )
    estimates = np.median(samples, axis=1)
    return tuple(np.quantile(estimates, (0.025, 0.975)).astype(float))


def _parse_surgery_events(raw, measurements):
    episode_map = measurements[
        [
            "hospital_id",
            "episode_id",
            "admission_time",
            "discharge_time",
        ]
    ].drop_duplicates()
    metadata_columns = [
        "hospital_id",
        "admission_time",
        "discharge_time",
        *SURGERY_COLUMNS,
    ]
    metadata = raw[metadata_columns].drop_duplicates().copy()
    metadata = metadata.merge(
        episode_map,
        on=["hospital_id", "admission_time", "discharge_time"],
        how="left",
        validate="one_to_one",
    )
    metadata["source_episode_key"] = (
        metadata["hospital_id"]
        + "|"
        + metadata["admission_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "|"
        + metadata["discharge_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    metadata = metadata.rename(
        columns={
            SURGERY_COLUMNS[0]: "surgery_starts_raw",
            SURGERY_COLUMNS[1]: "surgery_ends_raw",
            SURGERY_COLUMNS[2]: "surgery_names_raw",
        }
    )

    event_rows = []
    episode_rows = []
    for episode in metadata.itertuples(index=False):
        starts = _split_tokens(episode.surgery_starts_raw)
        ends = _split_tokens(episode.surgery_ends_raw)
        names = _split_tokens(episode.surgery_names_raw)
        token_count = max(len(starts), len(ends), len(names))
        episode_events = []
        for position in range(token_count):
            start_text = starts[position] if position < len(starts) else ""
            end_text = ends[position] if position < len(ends) else ""
            name = names[position] if position < len(names) else ""
            start = pd.to_datetime(start_text, errors="coerce")
            end = pd.to_datetime(end_text, errors="coerce")
            declared = bool(start_text or end_text or name)
            parsed_pair = pd.notna(start) and pd.notna(end)
            positive_duration = bool(parsed_pair and end > start)
            within_stay = bool(
                positive_duration
                and start >= episode.admission_time
                and end <= episode.discharge_time
            )
            valid = bool(declared and within_stay)
            reasons = []
            if declared and not parsed_pair:
                reasons.append("unparseable_or_missing_time")
            if parsed_pair and not positive_duration:
                reasons.append("nonpositive_duration")
            if positive_duration and not within_stay:
                reasons.append("outside_hospitalization")
            if not declared:
                reasons.append("no_surgery_record")
            row = {
                "source_episode_key": episode.source_episode_key,
                "episode_id": episode.episode_id,
                "hospital_id": episode.hospital_id,
                "admission_time": episode.admission_time,
                "discharge_time": episode.discharge_time,
                "surgery_position": position,
                "surgery_name": name,
                "surgery_start_text": start_text,
                "surgery_end_text": end_text,
                "surgery_start": start,
                "surgery_end": end,
                "declared_event": declared,
                "parsed_time_pair": parsed_pair,
                "positive_duration": positive_duration,
                "within_hospitalization": within_stay,
                "valid_surgery_event": valid,
                "is_cabg": bool(
                    valid and "冠状动脉旁路移植" in name
                ),
                "audit_reason": "valid" if valid else ";".join(reasons),
                "token_counts_match": (
                    len(starts) == len(ends) == len(names)
                ),
                "start_token_count": len(starts),
                "end_token_count": len(ends),
                "name_token_count": len(names),
            }
            if valid:
                row["duration_hours"] = (
                    end - start
                ).total_seconds() / 3600.0
                row["days_from_admission_to_start"] = (
                    start - episode.admission_time
                ).total_seconds() / 86400.0
                row["days_from_end_to_discharge"] = (
                    episode.discharge_time - end
                ).total_seconds() / 86400.0
            else:
                row["duration_hours"] = np.nan
                row["days_from_admission_to_start"] = np.nan
                row["days_from_end_to_discharge"] = np.nan
            episode_events.append(row)
            event_rows.append(row)

        valid_events = [row for row in episode_events if row["valid_surgery_event"]]
        valid_events.sort(key=lambda row: row["surgery_position"])
        cabg_events = [row for row in valid_events if row["is_cabg"]]
        principal = valid_events[0] if valid_events else None
        cabg = cabg_events[0] if cabg_events else None
        declared_count = sum(row["declared_event"] for row in episode_events)
        episode_rows.append(
            {
                "source_episode_key": episode.source_episode_key,
                "episode_id": episode.episode_id,
                "hospital_id": episode.hospital_id,
                "admission_time": episode.admission_time,
                "discharge_time": episode.discharge_time,
                "linked_to_clean_measurements": pd.notna(episode.episode_id),
                "token_counts_match": (
                    len(starts) == len(ends) == len(names)
                ),
                "declared_event_count": declared_count,
                "valid_event_count": len(valid_events),
                "invalid_declared_event_count": declared_count - len(valid_events),
                "has_valid_principal_surgery": principal is not None,
                "has_valid_cabg": cabg is not None,
                "surgery_record_status": (
                    "valid"
                    if valid_events
                    else ("invalid_only" if declared_count else "none")
                ),
                "principal_surgery_position": (
                    principal["surgery_position"] if principal else np.nan
                ),
                "principal_surgery_name": (
                    principal["surgery_name"] if principal else ""
                ),
                "principal_surgery_start": (
                    principal["surgery_start"] if principal else pd.NaT
                ),
                "principal_surgery_end": (
                    principal["surgery_end"] if principal else pd.NaT
                ),
                "cabg_surgery_position": (
                    cabg["surgery_position"] if cabg else np.nan
                ),
                "cabg_surgery_start": (
                    cabg["surgery_start"] if cabg else pd.NaT
                ),
                "cabg_surgery_end": (
                    cabg["surgery_end"] if cabg else pd.NaT
                ),
            }
        )

    events = pd.DataFrame(event_rows)
    episodes = pd.DataFrame(episode_rows)
    valid = events[events["valid_surgery_event"]].copy()
    principal_keys = set(
        zip(
            episodes.loc[
                episodes["has_valid_principal_surgery"],
                "source_episode_key",
            ],
            episodes.loc[
                episodes["has_valid_principal_surgery"],
                "principal_surgery_position",
            ].astype(int),
        )
    )
    cabg_keys = set(
        zip(
            episodes.loc[episodes["has_valid_cabg"], "source_episode_key"],
            episodes.loc[
                episodes["has_valid_cabg"], "cabg_surgery_position"
            ].astype(int),
        )
    )
    events["selected_principal_anchor"] = [
        (key, position) in principal_keys
        for key, position in zip(
            events["source_episode_key"], events["surgery_position"]
        )
    ]
    events["selected_cabg_anchor"] = [
        (key, position) in cabg_keys
        for key, position in zip(
            events["source_episode_key"], events["surgery_position"]
        )
    ]
    procedure_summary = (
        valid.groupby("surgery_name")
        .agg(
            valid_events=("surgery_name", "size"),
            episodes=("source_episode_key", "nunique"),
            patients=("hospital_id", "nunique"),
            median_duration_hours=("duration_hours", "median"),
        )
        .reset_index()
    )
    principal_counts = (
        events[events["selected_principal_anchor"]]
        .groupby("surgery_name")
        .size()
        .rename("principal_episode_count")
        .reset_index()
    )
    procedure_summary = (
        procedure_summary.merge(principal_counts, on="surgery_name", how="left")
        .fillna({"principal_episode_count": 0})
        .sort_values(["episodes", "valid_events"], ascending=False)
    )
    procedure_summary["principal_episode_count"] = procedure_summary[
        "principal_episode_count"
    ].astype(int)
    return events, episodes, procedure_summary


def _build_anchors(events):
    frames = []
    definitions = (
        ("cabg", "selected_cabg_anchor"),
        ("all_surgery", "selected_principal_anchor"),
    )
    for cohort, selector in definitions:
        selected = events[
            events[selector]
            & events["valid_surgery_event"]
            & events["episode_id"].notna()
        ].copy()
        selected["cohort"] = cohort
        frames.append(
            selected[
                [
                    "cohort",
                    "hospital_id",
                    "episode_id",
                    "surgery_name",
                    "surgery_start",
                    "surgery_end",
                    "duration_hours",
                    "days_from_admission_to_start",
                    "days_from_end_to_discharge",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _assign_phases(frame):
    report = frame["report_time"]
    start = frame["surgery_start"]
    end = frame["surgery_end"]
    pre_hours = (report - start).dt.total_seconds() / 3600.0
    post_hours = (report - end).dt.total_seconds() / 3600.0
    phase = np.full(len(frame), "", dtype=object)
    pre = report.lt(start)
    intra = report.ge(start) & report.le(end)
    post = report.gt(end)
    phase[pre & pre_hours.lt(-168)] = "pre_gt7d"
    phase[pre & pre_hours.ge(-168) & pre_hours.lt(-72)] = "pre_3_7d"
    phase[pre & pre_hours.ge(-72) & pre_hours.lt(-24)] = "pre_1_3d"
    phase[pre & pre_hours.ge(-24)] = "pre_lt24h"
    phase[intra] = "intraop"
    phase[post & post_hours.le(6)] = "post_0_6h"
    phase[post & post_hours.gt(6) & post_hours.le(24)] = "post_6_24h"
    phase[post & post_hours.gt(24) & post_hours.le(48)] = "post_1_2d"
    phase[post & post_hours.gt(48) & post_hours.le(72)] = "post_2_3d"
    phase[post & post_hours.gt(72) & post_hours.le(168)] = "post_3_7d"
    phase[post & post_hours.gt(168)] = "post_gt7d"
    if np.any(phase == ""):
        raise AssertionError("Some surgery-linked measurements were not phased")
    frame = frame.copy()
    frame["surgery_phase"] = phase
    frame["phase_order"] = frame["surgery_phase"].map(PHASE_ORDER).astype(int)
    frame["hours_from_surgery_start"] = pre_hours
    frame["hours_from_surgery_end"] = post_hours
    return frame


def _phase_statistics(measurements, anchors, dictionary):
    linked = measurements.merge(
        anchors,
        on=["hospital_id", "episode_id"],
        how="inner",
        validate="many_to_many",
    )
    linked = _assign_phases(linked)
    episode_values = (
        linked.groupby(
            [
                "cohort",
                "hospital_id",
                "episode_id",
                "variable_id",
                "surgery_phase",
                "phase_order",
            ],
            as_index=False,
        )
        .agg(
            episode_phase_value=("numeric_value", "median"),
            measurement_count=("numeric_value", "size"),
            median_hours_from_surgery_start=(
                "hours_from_surgery_start",
                "median",
            ),
        )
    )
    patient_values = (
        episode_values.groupby(
            ["cohort", "hospital_id", "variable_id", "surgery_phase", "phase_order"],
            as_index=False,
        )
        .agg(
            patient_phase_value=("episode_phase_value", "median"),
            episode_count=("episode_id", "nunique"),
            measurement_count=("measurement_count", "sum"),
        )
    )
    summary = (
        patient_values.groupby(
            ["cohort", "variable_id", "surgery_phase", "phase_order"]
        )
        .agg(
            patients=("hospital_id", "nunique"),
            median=("patient_phase_value", "median"),
            q25=("patient_phase_value", lambda x: np.quantile(x, 0.25)),
            q75=("patient_phase_value", lambda x: np.quantile(x, 0.75)),
        )
        .reset_index()
    )
    episode_coverage = (
        episode_values.groupby(
            ["cohort", "variable_id", "surgery_phase", "phase_order"]
        )
        .agg(
            episodes=("episode_id", "nunique"),
            measurements=("measurement_count", "sum"),
        )
        .reset_index()
    )
    summary = summary.merge(
        episode_coverage,
        on=["cohort", "variable_id", "surgery_phase", "phase_order"],
        how="left",
    )
    patient_global = (
        linked.groupby(["cohort", "variable_id", "hospital_id"], as_index=False)[
            "numeric_value"
        ]
        .median()
        .rename(columns={"numeric_value": "patient_global_value"})
    )
    scale = (
        patient_global.groupby(["cohort", "variable_id"])
        .agg(
            global_median=("patient_global_value", "median"),
            global_q25=("patient_global_value", lambda x: np.quantile(x, 0.25)),
            global_q75=("patient_global_value", lambda x: np.quantile(x, 0.75)),
        )
        .reset_index()
    )
    scale["global_iqr"] = scale["global_q75"] - scale["global_q25"]
    scale.loc[scale["global_iqr"].le(0), "global_iqr"] = np.nan
    summary = summary.merge(scale, on=["cohort", "variable_id"], how="left")
    summary["robust_z_median"] = (
        summary["median"] - summary["global_median"]
    ) / summary["global_iqr"]
    summary["phase_label"] = summary["surgery_phase"].map(PHASE_LABELS)
    summary = summary.merge(dictionary, on="variable_id", how="left")
    coverage = summary[
        [
            "cohort",
            "variable_id",
            "surgery_phase",
            "phase_order",
            "phase_label",
            "patients",
            "episodes",
            "measurements",
            "item_name_cn",
            "item_name_en",
            "unit",
        ]
    ].copy()
    return linked, episode_values, patient_values, summary, coverage


def _endpoint_values(linked):
    records = []
    group_columns = ["cohort", "hospital_id", "episode_id", "variable_id"]
    for keys, group in linked.groupby(group_columns, sort=True):
        cohort, hospital_id, episode_id, variable_id = keys
        group = group.sort_values("report_time")
        start = group["surgery_start"].iloc[0]
        end = group["surgery_end"].iloc[0]
        windows = {
            "last_preop_7d": group[
                group["report_time"].lt(start)
                & group["report_time"].ge(start - pd.Timedelta(days=7))
            ].tail(1),
            "intraop": group[
                group["report_time"].ge(start)
                & group["report_time"].le(end)
            ],
            "post_0_24h": group[
                group["report_time"].gt(end)
                & group["report_time"].le(end + pd.Timedelta(hours=24))
            ],
            "post_3_7d": group[
                group["report_time"].gt(end + pd.Timedelta(days=3))
                & group["report_time"].le(end + pd.Timedelta(days=7))
            ],
            "predischarge": group[group["report_time"].gt(end)].tail(1),
        }
        for endpoint, values in windows.items():
            if values.empty:
                continue
            if endpoint in {"last_preop_7d", "predischarge"}:
                value = float(values["numeric_value"].iloc[-1])
                timestamp = values["report_time"].iloc[-1]
            else:
                value = float(values["numeric_value"].median())
                timestamp = values["report_time"].sort_values().iloc[
                    len(values) // 2
                ]
            records.append(
                {
                    "cohort": cohort,
                    "hospital_id": hospital_id,
                    "episode_id": episode_id,
                    "variable_id": variable_id,
                    "endpoint": endpoint,
                    "endpoint_value": value,
                    "representative_time": timestamp,
                    "measurement_count": int(len(values)),
                    "hours_from_surgery_start": float(
                        (timestamp - start).total_seconds() / 3600.0
                    ),
                    "hours_from_surgery_end": float(
                        (timestamp - end).total_seconds() / 3600.0
                    ),
                }
            )
    return pd.DataFrame(records)


def _contrast_statistics(
    endpoint_values,
    measurements,
    dictionary,
    minimum_paired_patients,
    bootstrap_replicates,
    seed,
):
    lookup = (
        endpoint_values.pivot_table(
            index=["cohort", "hospital_id", "episode_id", "variable_id"],
            columns="endpoint",
            values="endpoint_value",
            aggfunc="first",
        )
        .reset_index()
    )
    episode_records = []
    for contrast, source, target in CONTRASTS:
        if source not in lookup or target not in lookup:
            continue
        available = lookup[lookup[source].notna() & lookup[target].notna()]
        for row in available.itertuples(index=False):
            source_value = float(getattr(row, source))
            target_value = float(getattr(row, target))
            episode_records.append(
                {
                    "cohort": row.cohort,
                    "hospital_id": row.hospital_id,
                    "episode_id": row.episode_id,
                    "variable_id": row.variable_id,
                    "contrast": contrast,
                    "source_endpoint": source,
                    "target_endpoint": target,
                    "source_value": source_value,
                    "target_value": target_value,
                    "absolute_change": target_value - source_value,
                }
            )
    episode_changes = pd.DataFrame(episode_records)
    patient_changes = (
        episode_changes.groupby(
            ["cohort", "contrast", "variable_id", "hospital_id"],
            as_index=False,
        )
        .agg(
            patient_median_change=("absolute_change", "median"),
            patient_median_source=("source_value", "median"),
            patient_median_target=("target_value", "median"),
            episode_count=("episode_id", "nunique"),
        )
    )
    measurement_iqr = (
        measurements.groupby("variable_id")["numeric_value"]
        .agg(
            q25=lambda x: np.quantile(x, 0.25),
            q75=lambda x: np.quantile(x, 0.75),
        )
        .reset_index()
    )
    measurement_iqr["measurement_iqr"] = (
        measurement_iqr["q75"] - measurement_iqr["q25"]
    )
    iqr_lookup = measurement_iqr.set_index("variable_id")[
        "measurement_iqr"
    ].to_dict()
    rng = np.random.default_rng(seed + 911)
    rows = []
    for keys, group in patient_changes.groupby(
        ["cohort", "contrast", "variable_id"], sort=True
    ):
        cohort, contrast, variable_id = keys
        if len(group) < minimum_paired_patients:
            continue
        values = group["patient_median_change"].to_numpy(np.float64)
        iqr = float(iqr_lookup[variable_id])
        scale_iqr = iqr if iqr > 0 else np.nan
        statistic, p_value = _safe_wilcoxon(values)
        ci_low, ci_high = _bootstrap_median_ci(
            values,
            rng,
            bootstrap_replicates,
        )
        rows.append(
            {
                "cohort": cohort,
                "contrast": contrast,
                "contrast_label": CONTRAST_LABELS[contrast],
                "variable_id": variable_id,
                "paired_patients": int(len(group)),
                "paired_episodes": int(
                    episode_changes[
                        episode_changes["cohort"].eq(cohort)
                        & episode_changes["contrast"].eq(contrast)
                        & episode_changes["variable_id"].eq(variable_id)
                    ]["episode_id"].nunique()
                ),
                "median_change": float(np.median(values)),
                "median_change_ci95_low": ci_low,
                "median_change_ci95_high": ci_high,
                "measurement_iqr": iqr,
                "standardized_median_change_iqr": float(
                    np.median(values) / scale_iqr
                ),
                "standardized_ci95_low": ci_low / scale_iqr,
                "standardized_ci95_high": ci_high / scale_iqr,
                "rank_biserial": _rank_biserial(values),
                "increase_patients": int(np.count_nonzero(values > 0)),
                "decrease_patients": int(np.count_nonzero(values < 0)),
                "unchanged_patients": int(
                    np.count_nonzero(np.isclose(values, 0))
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    statistics = pd.DataFrame(rows)
    if not statistics.empty:
        statistics["q_value_bh"] = np.nan
        for cohort, indices in statistics.groupby("cohort").groups.items():
            statistics.loc[indices, "q_value_bh"] = _benjamini_hochberg(
                statistics.loc[indices, "p_value"]
            )
        statistics["fdr_significant_0_05"] = statistics["q_value_bh"].le(0.05)
        statistics = statistics.merge(dictionary, on="variable_id", how="left")
    return episode_changes, patient_changes, statistics


def _plot_overview(events, episodes, procedure_summary, anchors, output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    counts = [
        len(episodes),
        int(episodes["declared_event_count"].gt(0).sum()),
        int(episodes["has_valid_principal_surgery"].sum()),
        int(episodes["has_valid_cabg"].sum()),
        int(
            episodes[
                episodes["has_valid_cabg"]
                & episodes["linked_to_clean_measurements"]
            ].shape[0]
        ),
    ]
    labels = [
        "Source episodes",
        "Surgery declared",
        "Valid principal",
        "Valid CABG",
        "CABG + clean labs",
    ]
    bars = axes[0, 0].barh(labels[::-1], counts[::-1], color="#2F6B8A")
    for bar, value in zip(bars, counts[::-1]):
        axes[0, 0].text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
        )
    axes[0, 0].set_title("Surgery cohort inclusion")
    axes[0, 0].set_xlabel("Hospital episodes")

    top = procedure_summary.head(12).sort_values("episodes")
    axes[0, 1].barh(
        top["surgery_name"].map(PROCEDURE_ENGLISH).fillna("Other procedure"),
        top["episodes"],
        color="#4D8C72",
    )
    axes[0, 1].set_title("Most frequent valid procedures")
    axes[0, 1].set_xlabel("Episodes")
    axes[0, 1].tick_params(axis="y", labelsize=7)

    for cohort, color in (("cabg", "#C46A45"), ("all_surgery", "#2F6B8A")):
        values = anchors.loc[
            anchors["cohort"].eq(cohort), "duration_hours"
        ].to_numpy(np.float64)
        axes[1, 0].hist(
            values,
            bins=np.linspace(0, min(12, max(8, math.ceil(values.max()))), 25),
            alpha=0.55,
            label=cohort.replace("_", " "),
            color=color,
        )
    axes[1, 0].set_xlabel("Duration (hours)")
    axes[1, 0].set_ylabel("Episodes")
    axes[1, 0].set_title("Selected surgery duration")
    axes[1, 0].legend()

    cabg = anchors[anchors["cohort"].eq("cabg")]
    axes[1, 1].scatter(
        cabg["days_from_admission_to_start"],
        cabg["days_from_end_to_discharge"],
        s=15,
        alpha=0.45,
        color="#6D597A",
    )
    axes[1, 1].set_xlabel("Admission to surgery start (days)")
    axes[1, 1].set_ylabel("Surgery end to discharge (days)")
    axes[1, 1].set_title("CABG timing within hospitalization")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path = os.path.join(output_dir, "surgery_cohort_overview.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_phase_coverage(linked, output_dir):
    any_phase = (
        linked.groupby(["cohort", "surgery_phase"])
        .agg(
            episodes=("episode_id", "nunique"),
            patients=("hospital_id", "nunique"),
            measurements=("numeric_value", "size"),
        )
        .reset_index()
    )
    any_phase["phase_order"] = any_phase["surgery_phase"].map(PHASE_ORDER)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(PHASES))
    width = 0.38
    for offset, (cohort, color) in zip(
        (-width / 2, width / 2),
        (("cabg", "#C46A45"), ("all_surgery", "#2F6B8A")),
    ):
        values = (
            any_phase[any_phase["cohort"].eq(cohort)]
            .set_index("surgery_phase")
            .reindex([phase for phase, _ in PHASES])
        )
        axes[0].bar(
            x + offset,
            values["episodes"].fillna(0),
            width,
            color=color,
            label=cohort.replace("_", " "),
        )
        axes[1].bar(
            x + offset,
            values["measurements"].fillna(0),
            width,
            color=color,
            label=cohort.replace("_", " "),
        )
    axes[0].set_ylabel("Episodes with any lab")
    axes[1].set_ylabel("Collapsed measurements")
    axes[1].set_xticks(x, [label for _, label in PHASES], rotation=30, ha="right")
    axes[0].set_title("Surgery-aligned laboratory coverage")
    for axis in axes:
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = os.path.join(output_dir, "surgery_phase_coverage.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return any_phase, path


def _eligible_phase_variables(summary, cohort="cabg", minimum_patients=20):
    selected = summary[summary["cohort"].eq(cohort)].copy()
    coverage = (
        selected.assign(adequate=selected["patients"].ge(minimum_patients))
        .groupby("variable_id")
        .agg(
            adequate_phases=("adequate", "sum"),
            max_patients=("patients", "max"),
            robust_range=(
                "robust_z_median",
                lambda x: (
                    float(np.nanmax(x) - np.nanmin(x))
                    if np.isfinite(x).any()
                    else np.nan
                ),
            ),
        )
    )
    return coverage[
        coverage["adequate_phases"].ge(3)
        & coverage["max_patients"].ge(minimum_patients)
        & coverage["robust_range"].notna()
    ].sort_values("robust_range", ascending=False)


def _plot_phase_heatmap(summary, output_dir):
    eligible = _eligible_phase_variables(summary)
    selected_ids = eligible.index.tolist()
    values = summary[
        summary["cohort"].eq("cabg")
        & summary["variable_id"].isin(selected_ids)
    ]
    matrix = (
        values.pivot(
            index="variable_id",
            columns="surgery_phase",
            values="robust_z_median",
        )
        .reindex(index=selected_ids, columns=[phase for phase, _ in PHASES])
    )
    dictionary = (
        values.drop_duplicates("variable_id").set_index("variable_id")
    )
    labels = [
        f"{variable_id} {dictionary.loc[variable_id, 'item_name_en']}"
        for variable_id in matrix.index
    ]
    figure, axis = plt.subplots(
        figsize=(14, max(7, 0.38 * max(1, len(matrix))))
    )
    limit = max(
        1.0,
        float(np.nanquantile(np.abs(matrix.to_numpy(np.float64)), 0.98)),
    )
    image = axis.imshow(
        matrix.to_numpy(np.float64),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(
        np.arange(len(PHASES)),
        [label for _, label in PHASES],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_title("CABG-aligned patient-balanced laboratory trajectories")
    colorbar = figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.set_label("Median level relative to patient-level global median (IQR)")
    figure.tight_layout()
    path = os.path.join(output_dir, "cabg_surgery_aligned_heatmap.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return selected_ids, path


def _plot_top_phase_trajectories(summary, selected_ids, output_dir):
    top_ids = selected_ids[:12]
    figure, axes = plt.subplots(4, 3, figsize=(16, 14), squeeze=False)
    for axis, variable_id in zip(axes.flat, top_ids):
        values = summary[
            summary["cohort"].eq("cabg")
            & summary["variable_id"].eq(variable_id)
        ].sort_values("phase_order")
        x = values["phase_order"].to_numpy()
        axis.plot(x, values["median"], color="#2F6B8A", marker="o")
        axis.fill_between(
            x,
            values["q25"],
            values["q75"],
            color="#8CB7C9",
            alpha=0.35,
        )
        for point_x, point_y, patients in zip(
            x,
            values["median"],
            values["patients"],
        ):
            axis.annotate(
                f"n={int(patients)}",
                (point_x, point_y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5,
                color="#244F65",
            )
        row = values.iloc[0]
        axis.set_title(
            f"{variable_id} {row['item_name_en']}",
            fontsize=9,
        )
        axis.set_ylabel(row["unit"])
        axis.margins(y=0.12)
        axis.set_xticks(
            range(len(PHASES)),
            [label for _, label in PHASES],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(top_ids):]:
        axis.axis("off")
    figure.suptitle(
        "Largest CABG-aligned changes: patient-balanced median and IQR",
        fontsize=14,
    )
    figure.tight_layout()
    path = os.path.join(output_dir, "cabg_surgery_trajectory_panels.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_contrast_forest(statistics, output_dir):
    selected = statistics[
        statistics["cohort"].eq("cabg")
        & np.isfinite(statistics["standardized_median_change_iqr"])
        & np.isfinite(statistics["standardized_ci95_low"])
        & np.isfinite(statistics["standardized_ci95_high"])
    ].copy()
    selected["magnitude"] = selected["standardized_median_change_iqr"].abs()
    panels = []
    for contrast, _, _ in CONTRASTS:
        panel = selected[selected["contrast"].eq(contrast)].copy()
        panel = panel.nlargest(8, "magnitude")
        panels.append(panel.sort_values("standardized_median_change_iqr"))

    displayed = pd.concat(panels, ignore_index=True)
    lower_limit = float(displayed["standardized_ci95_low"].min())
    upper_limit = float(displayed["standardized_ci95_high"].max())
    span = max(0.5, upper_limit - lower_limit)
    x_limits = (lower_limit - 0.05 * span, upper_limit + 0.05 * span)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(18, 13),
        sharex=True,
        squeeze=False,
    )
    decrease_color = "#2F6B8A"
    increase_color = "#C46A45"
    for axis, (contrast, _, _), panel in zip(
        axes.flat,
        CONTRASTS,
        panels,
    ):
        y = np.arange(len(panel))
        for index, row in enumerate(panel.itertuples(index=False)):
            value = row.standardized_median_change_iqr
            color = increase_color if value >= 0 else decrease_color
            axis.errorbar(
                value,
                index,
                xerr=np.array(
                    [
                        [value - row.standardized_ci95_low],
                        [row.standardized_ci95_high - value],
                    ]
                ),
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=2.5,
            )
            axis.scatter(
                value,
                index,
                s=42,
                facecolor=color if row.fdr_significant_0_05 else "white",
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
        labels = [
            f"{row.variable_id} {row.item_name_en} (n={row.paired_patients})"
            for row in panel.itertuples(index=False)
        ]
        axis.set_yticks(y, labels, fontsize=8)
        axis.set_xlim(*x_limits)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_title(CONTRAST_LABELS[contrast], fontsize=11)
        axis.grid(axis="x", alpha=0.2)

    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=decrease_color,
            markerfacecolor=decrease_color,
            linewidth=1.5,
            label="Decrease",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=increase_color,
            markerfacecolor=increase_color,
            linewidth=1.5,
            label="Increase",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#555555",
            markerfacecolor="#555555",
            linewidth=0,
            label="BH-FDR q < 0.05",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#555555",
            markerfacecolor="white",
            linewidth=0,
            label="Not FDR-significant",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "CABG-aligned paired laboratory changes by clinical interval",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    figure.supxlabel(
        "Median paired change / global measurement IQR "
        "(patient bootstrap 95% CI)",
        y=0.035,
    )
    figure.text(
        0.5,
        0.012,
        "Eight largest absolute standardized changes are shown per interval.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 0.925), h_pad=2.3, w_pad=2.0)
    path = os.path.join(output_dir, "cabg_surgery_contrast_forest.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_sensitivity(statistics, output_dir):
    values = statistics.pivot_table(
        index=["contrast", "variable_id"],
        columns="cohort",
        values="standardized_median_change_iqr",
        aggfunc="first",
    ).dropna()
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), squeeze=False)
    for axis, (contrast, _, _) in zip(axes.flat, CONTRASTS):
        subset = values.loc[
            values.index.get_level_values("contrast") == contrast
        ]
        if len(subset):
            x = subset["all_surgery"].to_numpy(np.float64)
            y = subset["cabg"].to_numpy(np.float64)
            axis.scatter(x, y, color="#2F6B8A", alpha=0.7)
            limit = max(0.5, float(np.max(np.abs(np.concatenate((x, y))))))
            axis.plot((-limit, limit), (-limit, limit), "--", color="#777777")
            axis.set_xlim(-limit * 1.05, limit * 1.05)
            axis.set_ylim(-limit * 1.05, limit * 1.05)
        axis.set_title(CONTRAST_LABELS[contrast])
        axis.set_xlabel("All valid principal surgeries")
        axis.set_ylabel("CABG")
        axis.grid(alpha=0.2)
    figure.suptitle("Sensitivity of standardized paired changes", fontsize=14)
    figure.tight_layout()
    path = os.path.join(output_dir, "surgery_all_vs_cabg_sensitivity.png")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return values.reset_index(), path


def _plot_phase_pdf(summary, selected_ids, output_dir):
    path = os.path.join(output_dir, "all_surgery_aligned_trajectories.pdf")
    with PdfPages(path) as pdf:
        for variable_id in sorted(selected_ids):
            figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            for axis, cohort in zip(axes, ("cabg", "all_surgery")):
                values = summary[
                    summary["cohort"].eq(cohort)
                    & summary["variable_id"].eq(variable_id)
                ].sort_values("phase_order")
                if values.empty:
                    axis.axis("off")
                    continue
                x = values["phase_order"].to_numpy()
                axis.plot(x, values["median"], color="#2F6B8A", marker="o")
                axis.fill_between(
                    x,
                    values["q25"],
                    values["q75"],
                    color="#8CB7C9",
                    alpha=0.35,
                )
                axis.set_xticks(
                    range(len(PHASES)),
                    [label for _, label in PHASES],
                    rotation=55,
                    ha="right",
                    fontsize=7,
                )
                axis.set_ylabel(values.iloc[0]["unit"])
                axis.set_title(cohort.replace("_", " "))
                axis.grid(alpha=0.2)
            first = summary[summary["variable_id"].eq(variable_id)].iloc[0]
            figure.suptitle(
                f"{variable_id} {first['item_name_en']} [{first['unit']}]",
                fontsize=13,
            )
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return path


def _format_quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _write_surgery_report(
    report_dir,
    events,
    episodes,
    anchors,
    any_phase_coverage,
    statistics,
):
    cabg = anchors[anchors["cohort"].eq("cabg")]
    all_surgery = anchors[anchors["cohort"].eq("all_surgery")]
    cabg_stats = statistics[statistics["cohort"].eq("cabg")].copy()
    significant = cabg_stats[cabg_stats["fdr_significant_0_05"]].copy()
    top = (
        significant.assign(
            magnitude=significant["standardized_median_change_iqr"].abs()
        )
        .sort_values("magnitude", ascending=False)
        .head(12)
    )
    rows = []
    for row in top.itertuples(index=False):
        rows.append(
            f"| {row.variable_id} | {row.item_name_cn} | "
            f"{row.contrast_label} | {row.median_change:.4g} | "
            f"{row.standardized_median_change_iqr:.3f} | "
            f"{row.paired_patients} | {row.q_value_bh:.3g} |"
        )
    if not rows:
        rows = ["| - | - | - | - | - | - | - |"]
    coverage_rows = []
    for phase, label in PHASES:
        values = any_phase_coverage[
            any_phase_coverage["cohort"].eq("cabg")
            & any_phase_coverage["surgery_phase"].eq(phase)
        ]
        episodes_count = int(values["episodes"].iloc[0]) if len(values) else 0
        measurements_count = (
            int(values["measurements"].iloc[0]) if len(values) else 0
        )
        coverage_rows.append(
            f"| {label} | {episodes_count} | {measurements_count:,} |"
        )
    duration = _format_quantiles(cabg["duration_hours"])
    admission_offset = _format_quantiles(
        cabg["days_from_admission_to_start"]
    )
    discharge_offset = _format_quantiles(
        cabg["days_from_end_to_discharge"]
    )
    text = f"""# 手术对齐化验值统计报告

## 手术队列

- 源表住院 episode：{len(episodes)} 次
- 有手术字段记录：{int(episodes['declared_event_count'].gt(0).sum())} 次
- 有至少一个时间有效且位于住院期内的手术：{int(episodes['has_valid_principal_surgery'].sum())} 次
- 有有效 CABG 时间：{int(episodes['has_valid_cabg'].sum())} 次
- 可关联清洗后化验的 CABG episode：{cabg['episode_id'].nunique()} 次，{cabg['hospital_id'].nunique()} 人
- 可关联清洗后化验的全部主手术 episode：{all_surgery['episode_id'].nunique()} 次，{all_surgery['hospital_id'].nunique()} 人
- 手术条目总数：{len(events)}；有效条目：{int(events['valid_surgery_event'].sum())}
- 手术字段 token 数不一致的 episode：{int((~episodes['token_counts_match']).sum())} 次

含多个手术的字段按 `^` 拆分，并严格按位置配对手术名称、开始时间和结束时间。主要分析以有效 CABG 条目为锚点；敏感性分析使用每次住院中列表位置最靠前的有效手术。开始/结束时间无法解析、结束不晚于开始或位于住院期外的条目不用于对齐，但保留在审计表中。

CABG 手术时长中位数为 {duration['median']:.2f} 小时（IQR {duration['q25']:.2f}-{duration['q75']:.2f}）；入院至手术开始中位数为 {admission_offset['median']:.2f} 天（IQR {admission_offset['q25']:.2f}-{admission_offset['q75']:.2f}）；手术结束至出院中位数为 {discharge_offset['median']:.2f} 天（IQR {discharge_offset['q25']:.2f}-{discharge_offset['q75']:.2f}）。

## CABG 分期覆盖

| 手术相对阶段 | 有任一化验的 episode | 清洗后测量 |
|---|---:|---:|
{chr(10).join(coverage_rows)}

每个非重叠阶段先在同一 episode 内取中位数，再在同一患者的重复住院间取中位数，最后计算患者间中位数和 IQR，避免化验频率较高的患者占更大权重。

输入化验值继承住院期主分析的字段规范化：血气葡萄糖单位、动脉/肺泡氧比值尺度及 P50 设备别名在进入手术分期前已经统一；标准条件与患者条件 P50 仍保持分离。规则和证据分别见 `../tables/variable_harmonization_audit.csv` 与 `../tables/field_equivalence_evidence.csv`。

## CABG 主要配对变化

| ID | 检验项 | 对比 | 中位变化 | 变化/IQR | 患者 | BH q |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

术前基线定义为手术开始前 7 天内最近一次有效结果。术中与术后 0-24 小时端点取窗口内中位数，术后 3-7 天取窗口中位数，出院前端点取手术后最后一次有效结果。变化先在 episode 内计算，再对同一患者取中位数。双侧 Wilcoxon 符号秩检验仅纳入至少 20 名配对患者的“检验项×对比”，并在每个手术队列内对全部检验进行 BH-FDR 校正；95% CI 为患者级 bootstrap 中位数区间。

## 解释限制

1. 手术相对时间揭示时间关联，不能单独证明手术造成化验变化。
2. 围手术期采样由临床需要决定，不同阶段的缺失不是随机缺失。
3. “全部手术”敏感性队列以首页列表中最靠前的有效手术为锚点；不同术式的临床过程异质，因此 CABG 队列是主要结果。
4. 术前 24 小时覆盖较低，正式配对基线扩展为术前 7 天内最近值；每个对比的实际配对患者数保存在统计表中。
5. 仅合并主分析审计表中有证据支持的单位或设备别名；其余不同名称或单位保持独立。数值方向不自动表示临床改善或恶化。
6. 全局 IQR 为 0 的检验项保留原始变化和显著性检验，但不计算“变化/IQR”，也不进入按标准化幅度排序的图。

## 图表与机器可读结果

- `../figures/surgery_cohort_overview.png`
- `../figures/surgery_phase_coverage.png`
- `../figures/cabg_surgery_aligned_heatmap.png`
- `../figures/cabg_surgery_trajectory_panels.png`
- `../figures/cabg_surgery_contrast_forest.png`
- `../figures/surgery_all_vs_cabg_sensitivity.png`
- `../figures/all_surgery_aligned_trajectories.pdf`
- `../tables/surgery_event_audit.csv`
- `../tables/surgery_episode_audit.csv`
- `../tables/surgery_procedure_summary.csv`
- `../tables/surgery_phase_episode_values.csv`
- `../tables/surgery_phase_patient_values.csv`
- `../tables/surgery_phase_summary.csv`
- `../tables/surgery_phase_coverage.csv`
- `../tables/surgery_endpoint_episode_values.csv`
- `../tables/surgery_contrast_episode_changes.csv`
- `../tables/surgery_contrast_patient_changes.csv`
- `../tables/surgery_contrast_statistics.csv`
- `../tables/surgery_sensitivity.csv`
- `../metadata/surgery_analysis_manifest.json`
"""
    with open(
        os.path.join(report_dir, "SURGERY_REPORT.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)
    return {
        "source_episodes": int(len(episodes)),
        "valid_principal_surgery_episodes": int(
            episodes["has_valid_principal_surgery"].sum()
        ),
        "valid_cabg_episodes": int(episodes["has_valid_cabg"].sum()),
        "linked_cabg_episodes": int(cabg["episode_id"].nunique()),
        "linked_cabg_patients": int(cabg["hospital_id"].nunique()),
        "linked_all_surgery_episodes": int(
            all_surgery["episode_id"].nunique()
        ),
        "cabg_duration_hours": duration,
        "cabg_days_admission_to_start": admission_offset,
        "cabg_days_end_to_discharge": discharge_offset,
        "cabg_tested_contrasts": int(len(cabg_stats)),
        "cabg_fdr_significant_contrasts": int(
            cabg_stats["fdr_significant_0_05"].sum()
        ),
        "report": "reports/SURGERY_REPORT.md",
    }


def run_surgery_analysis(
    raw,
    measurements,
    dictionary,
    output_dir,
    minimum_paired_patients,
    bootstrap_replicates,
    seed,
):
    figure_dir = os.path.join(output_dir, "figures")
    table_dir = os.path.join(output_dir, "tables")
    report_dir = os.path.join(output_dir, "reports")
    metadata_dir = os.path.join(output_dir, "metadata")
    for path in (figure_dir, table_dir, report_dir, metadata_dir):
        os.makedirs(path, exist_ok=True)
    events, episodes, procedure_summary = _parse_surgery_events(
        raw,
        measurements,
    )
    anchors = _build_anchors(events)
    (
        linked,
        episode_phase_values,
        patient_phase_values,
        phase_summary,
        phase_coverage,
    ) = _phase_statistics(measurements, anchors, dictionary)
    endpoint_values = _endpoint_values(linked)
    (
        contrast_episode_changes,
        contrast_patient_changes,
        contrast_statistics,
    ) = _contrast_statistics(
        endpoint_values,
        measurements,
        dictionary,
        minimum_paired_patients,
        bootstrap_replicates,
        seed,
    )

    events.to_csv(os.path.join(table_dir, "surgery_event_audit.csv"), index=False)
    episodes.to_csv(
        os.path.join(table_dir, "surgery_episode_audit.csv"),
        index=False,
    )
    procedure_summary.to_csv(
        os.path.join(table_dir, "surgery_procedure_summary.csv"),
        index=False,
    )
    episode_phase_values.to_csv(
        os.path.join(table_dir, "surgery_phase_episode_values.csv"),
        index=False,
    )
    patient_phase_values.to_csv(
        os.path.join(table_dir, "surgery_phase_patient_values.csv"),
        index=False,
    )
    phase_summary.to_csv(
        os.path.join(table_dir, "surgery_phase_summary.csv"),
        index=False,
    )
    phase_coverage.to_csv(
        os.path.join(table_dir, "surgery_phase_coverage.csv"),
        index=False,
    )
    endpoint_values.to_csv(
        os.path.join(table_dir, "surgery_endpoint_episode_values.csv"),
        index=False,
    )
    contrast_episode_changes.to_csv(
        os.path.join(table_dir, "surgery_contrast_episode_changes.csv"),
        index=False,
    )
    contrast_patient_changes.to_csv(
        os.path.join(table_dir, "surgery_contrast_patient_changes.csv"),
        index=False,
    )
    contrast_statistics.to_csv(
        os.path.join(table_dir, "surgery_contrast_statistics.csv"),
        index=False,
    )

    overview_path = _plot_overview(
        events,
        episodes,
        procedure_summary,
        anchors,
        figure_dir,
    )
    any_phase_coverage, coverage_path = _plot_phase_coverage(
        linked,
        figure_dir,
    )
    selected_ids, heatmap_path = _plot_phase_heatmap(
        phase_summary,
        figure_dir,
    )
    panels_path = _plot_top_phase_trajectories(
        phase_summary,
        selected_ids,
        figure_dir,
    )
    forest_path = _plot_contrast_forest(
        contrast_statistics,
        figure_dir,
    )
    sensitivity, sensitivity_path = _plot_sensitivity(
        contrast_statistics,
        figure_dir,
    )
    sensitivity.to_csv(
        os.path.join(table_dir, "surgery_sensitivity.csv"),
        index=False,
    )
    pdf_path = _plot_phase_pdf(phase_summary, selected_ids, figure_dir)

    summary = _write_surgery_report(
        report_dir,
        events,
        episodes,
        anchors,
        any_phase_coverage,
        contrast_statistics,
    )
    status_counts = Counter(episodes["surgery_record_status"])
    manifest = {
        "schema_version": 1,
        "primary_cohort": (
            "episodes with a valid CABG event; the first listed valid CABG "
            "event is the surgery anchor"
        ),
        "sensitivity_cohort": (
            "episodes with any valid surgery; the lowest-position valid "
            "procedure is the surgery anchor"
        ),
        "multi_value_parsing": (
            "split name/start/end fields by ^ and pair strictly by position"
        ),
        "valid_event": (
            "parseable start/end, end after start, and complete interval "
            "inside hospitalization"
        ),
        "phases": [
            {"phase": phase, "label": label, "order": PHASE_ORDER[phase]}
            for phase, label in PHASES
        ],
        "endpoints": {
            "last_preop_7d": "last value in [surgery start - 7d, start)",
            "intraop": "median value in [start, end]",
            "post_0_24h": "median value in (end, end + 24h]",
            "post_3_7d": "median value in (end + 3d, end + 7d]",
            "predischarge": "last value after surgery end and before discharge",
        },
        "contrasts": [
            {"contrast": name, "source": source, "target": target}
            for name, source, target in CONTRASTS
        ],
        "statistics": {
            "unit_of_inference": (
                "one median episode-level paired change per patient"
            ),
            "test": "two-sided Wilcoxon signed-rank",
            "multiple_testing": (
                "Benjamini-Hochberg across analyte-contrast tests within cohort"
            ),
            "confidence_interval": (
                f"{bootstrap_replicates}-replicate patient bootstrap median 95% CI"
            ),
            "minimum_paired_patients": minimum_paired_patients,
        },
        "counts": {
            **summary,
            "surgery_record_status": dict(status_counts),
            "valid_surgery_events": int(events["valid_surgery_event"].sum()),
            "phase_eligible_variables": int(len(selected_ids)),
        },
        "plots": [
            os.path.join("figures", os.path.basename(path))
            for path in (
                overview_path,
                coverage_path,
                heatmap_path,
                panels_path,
                forest_path,
                sensitivity_path,
                pdf_path,
            )
        ],
    }
    with open(
        os.path.join(metadata_dir, "surgery_analysis_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return summary
