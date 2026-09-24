"""Evaluate whether predicted current labs move up or down from the latest history."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .config import TARGETS
from .plot_results import TASK_LABELS, TASK_UNITS


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs/20frame"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "change_direction_analysis"
BOOTSTRAP_SEED = 20260921
BOOTSTRAP_REPLICATES = 2000

COLORS = (
    "#2F6B8A",
    "#C44E52",
    "#4C956C",
    "#D18F2F",
    "#7A5195",
    "#8C6D5A",
    "#3F8E8E",
    "#A65F46",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_correlation(function, first, second):
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return np.nan
    return float(function(first, second).statistic)


def _direction_metrics(frame):
    true = frame["true_increase"].to_numpy(np.uint8)
    predicted = frame["predicted_increase"].to_numpy(np.uint8)
    score = frame["predicted_delta"].to_numpy(np.float64)
    true_delta = frame["true_delta"].to_numpy(np.float64)
    predicted_delta = frame["predicted_delta"].to_numpy(np.float64)
    result = {
        "n": int(len(frame)),
        "patients": int(frame["hospital_id"].nunique()),
        "decreases": int((true == 0).sum()),
        "increases": int((true == 1).sum()),
        "increase_fraction": float(true.mean()) if len(true) else np.nan,
        "accuracy": float(accuracy_score(true, predicted)) if len(true) else np.nan,
        "balanced_accuracy": (
            float(balanced_accuracy_score(true, predicted))
            if len(np.unique(true)) == 2
            else np.nan
        ),
        "roc_auc": (
            float(roc_auc_score(true, score))
            if len(np.unique(true)) == 2
            else np.nan
        ),
        "macro_f1": (
            float(f1_score(true, predicted, average="macro", zero_division=0))
            if len(true)
            else np.nan
        ),
        "mcc": (
            float(matthews_corrcoef(true, predicted))
            if len(np.unique(true)) == 2
            else np.nan
        ),
        "delta_mae": float(np.mean(np.abs(predicted_delta - true_delta))),
        "delta_pearson_r": _safe_correlation(pearsonr, true_delta, predicted_delta),
        "delta_spearman_r": _safe_correlation(spearmanr, true_delta, predicted_delta),
    }
    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    result.update(
        {
            "true_decrease_pred_decrease": int(matrix[0, 0]),
            "true_decrease_pred_increase": int(matrix[0, 1]),
            "true_increase_pred_decrease": int(matrix[1, 0]),
            "true_increase_pred_increase": int(matrix[1, 1]),
            "decrease_recall": (
                float(matrix[0, 0] / matrix[0].sum())
                if matrix[0].sum()
                else np.nan
            ),
            "increase_recall": (
                float(matrix[1, 1] / matrix[1].sum())
                if matrix[1].sum()
                else np.nan
            ),
        }
    )
    return result


def _cluster_bootstrap(frame, replicates, seed):
    patient_groups = {
        patient: group.index.to_numpy()
        for patient, group in frame.groupby("hospital_id", sort=False)
    }
    patients = np.asarray(list(patient_groups), dtype=object)
    rng = np.random.default_rng(seed)
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "macro_f1",
        "mcc",
        "delta_pearson_r",
        "delta_spearman_r",
    )
    samples = {name: [] for name in metric_names}
    for _ in range(replicates):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([patient_groups[patient] for patient in selected])
        values = _direction_metrics(frame.loc[indices])
        for name in metric_names:
            if np.isfinite(values[name]):
                samples[name].append(values[name])
    intervals = {}
    for name, values in samples.items():
        intervals[f"{name}_ci_low"] = (
            float(np.quantile(values, 0.025)) if values else np.nan
        )
        intervals[f"{name}_ci_high"] = (
            float(np.quantile(values, 0.975)) if values else np.nan
        )
        intervals[f"{name}_bootstrap_valid"] = len(values)
    return intervals


def _latest_history(history):
    latest_times = history.groupby("video_id")["history_lab_time_unix"].transform("max")
    latest = history.loc[history["history_lab_time_unix"].eq(latest_times)].copy()
    ambiguity = (
        latest.groupby("video_id")["history_value"]
        .nunique(dropna=True)
        .rename("latest_distinct_values")
    )
    latest = (
        latest.sort_values(
            ["video_id", "history_lab_time_unix", "history_index_oldest_first"]
        )
        .drop_duplicates("video_id", keep="last")
        [[
            "video_id",
            "history_lab_time_unix",
            "history_value",
            "history_item_name",
            "history_unit",
            "history_minus_current_hours",
        ]]
        .merge(ambiguity, on="video_id", how="left", validate="one_to_one")
    )
    return latest


def _run_dir(input_dir, target, prediction_architecture):
    base = input_dir / "runs"
    return (
        base / prediction_architecture / target
        if prediction_architecture
        else base / target
    )


def _load_target(input_dir, history_dir, target, prediction_architecture):
    run_dir = _run_dir(input_dir, target, prediction_architecture)
    predictions = pd.read_csv(
        run_dir / "video_predictions.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    predictions = predictions.loc[predictions["split"].eq("test")].copy()
    predictions = predictions.drop(columns=["history_count"], errors="ignore")
    records = pd.read_csv(
        input_dir / "task_records" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str, "source_sample_id": str},
    )
    summaries = pd.read_csv(
        history_dir / "history_records" / f"{target}_summary.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    history = pd.read_csv(
        history_dir / "history_records" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    frame = predictions.merge(
        records[["video_id", "source_sample_id", "raw_value"]].rename(
            columns={"raw_value": "recorded_current_value"}
        ),
        on="video_id",
        how="left",
        validate="one_to_one",
    ).merge(
        summaries[[
            "video_id",
            "current_lab_time_unix",
            "history_count",
            "has_history",
            "history_unavailable_reason",
            "nearest_history_delta_hours",
        ]],
        on="video_id",
        how="left",
        validate="one_to_one",
    )
    if frame["source_sample_id"].isna().any():
        raise AssertionError(f"Missing source sample IDs for {target}")
    if not np.allclose(
        frame["raw_value"], frame["recorded_current_value"], rtol=0.0, atol=1e-9
    ):
        raise AssertionError(f"Prediction/record raw values differ for {target}")

    frame = frame.merge(
        _latest_history(history), on="video_id", how="left", validate="one_to_one"
    )
    frame["target"] = target
    frame["unit"] = TASK_UNITS[target]
    frame["exclusion_reason"] = ""
    no_history = frame["history_count"].fillna(0).eq(0) | frame["history_value"].isna()
    ambiguous = frame["latest_distinct_values"].fillna(0).gt(1)
    frame.loc[no_history, "exclusion_reason"] = "no_strictly_prior_same_episode_value"
    frame.loc[ambiguous, "exclusion_reason"] = "conflicting_values_at_latest_history_time"

    eligible = frame.loc[frame["exclusion_reason"].eq("")].copy()
    eligible["true_current_value"] = eligible["recorded_current_value"]
    eligible["predicted_current_value"] = eligible["y_pred"]
    eligible["previous_value"] = eligible["history_value"]
    eligible["true_delta"] = eligible["true_current_value"] - eligible["previous_value"]
    eligible["predicted_delta"] = (
        eligible["predicted_current_value"] - eligible["previous_value"]
    )
    true_tie = np.isclose(eligible["true_delta"], 0.0, rtol=0.0, atol=1e-9)
    eligible["direction_evaluable"] = ~true_tie
    eligible["true_direction"] = np.where(
        true_tie, "unchanged", np.where(eligible["true_delta"].gt(0), "increase", "decrease")
    )
    predicted_tie = np.isclose(eligible["predicted_delta"], 0.0, rtol=0.0, atol=1e-12)
    eligible["predicted_direction"] = np.where(
        predicted_tie,
        "unchanged",
        np.where(eligible["predicted_delta"].gt(0), "increase", "decrease"),
    )
    eligible["true_increase"] = eligible["true_delta"].gt(0).astype(np.uint8)
    eligible["predicted_increase"] = eligible["predicted_delta"].gt(0).astype(np.uint8)
    eligible["direction_correct"] = (
        eligible["true_increase"].eq(eligible["predicted_increase"])
        & eligible["direction_evaluable"]
    )
    eligible["previous_to_current_hours"] = -eligible["history_minus_current_hours"]
    return frame, eligible


def _aggregate_events(frame):
    key = ["target", "hospital_id", "source_sample_id"]
    consistency_columns = (
        "true_current_value",
        "previous_value",
        "current_lab_time_unix",
        "history_lab_time_unix",
    )
    for column in consistency_columns:
        if frame.groupby(key)[column].nunique(dropna=False).gt(1).any():
            raise AssertionError(f"Event-level {column} is inconsistent")
    event = frame.groupby(key, as_index=False).agg(
        split=("split", "first"),
        unit=("unit", "first"),
        video_count=("video_id", "nunique"),
        video_ids=("video_id", lambda values: "|".join(sorted(set(values)))),
        history_count=("history_count", "max"),
        current_lab_time_unix=("current_lab_time_unix", "first"),
        history_lab_time_unix=("history_lab_time_unix", "first"),
        previous_to_current_hours=("previous_to_current_hours", "first"),
        previous_value=("previous_value", "first"),
        true_current_value=("true_current_value", "first"),
        predicted_current_value=("predicted_current_value", "mean"),
    )
    event["true_delta"] = event["true_current_value"] - event["previous_value"]
    event["predicted_delta"] = event["predicted_current_value"] - event["previous_value"]
    event["direction_evaluable"] = ~np.isclose(
        event["true_delta"], 0.0, rtol=0.0, atol=1e-9
    )
    event["true_direction"] = np.where(
        ~event["direction_evaluable"],
        "unchanged",
        np.where(event["true_delta"].gt(0), "increase", "decrease"),
    )
    event["predicted_direction"] = np.where(
        np.isclose(event["predicted_delta"], 0.0, rtol=0.0, atol=1e-12),
        "unchanged",
        np.where(event["predicted_delta"].gt(0), "increase", "decrease"),
    )
    event["true_increase"] = event["true_delta"].gt(0).astype(np.uint8)
    event["predicted_increase"] = event["predicted_delta"].gt(0).astype(np.uint8)
    event["direction_correct"] = (
        event["true_increase"].eq(event["predicted_increase"])
        & event["direction_evaluable"]
    )
    return event


def _plot_performance(metrics, output_dir, model_label):
    target_metrics = metrics.loc[metrics["target"].isin(TARGETS)].set_index("target")
    targets = [target for target in TARGETS if target in target_metrics.index]
    y = np.arange(len(targets))
    figure, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    specifications = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC AUC"),
        ("accuracy", "Accuracy"),
    )
    for axis, (metric, title) in zip(axes, specifications):
        values = target_metrics.loc[targets, metric].to_numpy(float)
        low = target_metrics.loc[targets, f"{metric}_ci_low"].to_numpy(float)
        high = target_metrics.loc[targets, f"{metric}_ci_high"].to_numpy(float)
        errors = np.vstack((values - low, high - values))
        for index, (value, color) in enumerate(zip(values, COLORS)):
            axis.errorbar(
                value,
                y[index],
                xerr=errors[:, index:index + 1],
                fmt="o",
                ms=7,
                capsize=3,
                color=color,
            )
            axis.text(value + 0.015, y[index], f"{value:.2f}", va="center", fontsize=8)
        axis.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Estimate with patient-clustered 95% CI", xlim=(0, 1))
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [TASK_LABELS[target] for target in targets])
    axes[0].invert_yaxis()
    figure.suptitle(
        f"{model_label}: test direction relative to latest prior lab value"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "direction_performance.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_confusions(events, output_dir, model_label):
    targets = [target for target in TARGETS if target in set(events["target"])]
    figure, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for axis, target in zip(axes.flat, targets):
        selected = events.loc[
            events["target"].eq(target) & events["direction_evaluable"]
        ]
        matrix = confusion_matrix(
            selected["true_increase"], selected["predicted_increase"], labels=[0, 1]
        )
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{normalized[row, column]:.2f}\n(n={matrix[row, column]})",
                    ha="center",
                    va="center",
                    color="white" if normalized[row, column] > 0.55 else "#222222",
                    fontsize=9,
                )
        axis.set_xticks((0, 1), ("Decrease", "Increase"))
        axis.set_yticks((0, 1), ("Decrease", "Increase"))
        axis.set_xlabel("Predicted direction")
        axis.set_ylabel("True direction")
        axis.set_title(f"{TASK_LABELS[target]} | n={len(selected)}")
    figure.suptitle(f"{model_label}: direction confusion matrices", y=0.99)
    figure.subplots_adjust(
        left=0.06, right=0.98, bottom=0.08, top=0.91, wspace=0.35, hspace=0.38
    )
    figure.savefig(output_dir / "direction_confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_deltas(events, output_dir, model_label):
    targets = [target for target in TARGETS if target in set(events["target"])]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, target, color in zip(axes.flat, targets, COLORS):
        selected = events.loc[
            events["target"].eq(target) & events["direction_evaluable"]
        ].copy()
        limit = float(
            np.quantile(
                np.abs(np.concatenate((selected.true_delta, selected.predicted_delta))),
                0.98,
            )
        )
        limit = max(limit, 1e-9)
        correct = selected["direction_correct"].to_numpy(bool)
        axis.scatter(
            selected.loc[correct, "true_delta"],
            selected.loc[correct, "predicted_delta"],
            s=16,
            alpha=0.55,
            color=color,
            label="Correct direction",
        )
        axis.scatter(
            selected.loc[~correct, "true_delta"],
            selected.loc[~correct, "predicted_delta"],
            s=19,
            alpha=0.7,
            color="#C44E52",
            marker="x",
            label="Wrong direction",
        )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.plot((-limit, limit), (-limit, limit), linestyle=":", color="#777777")
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit))
        axis.set_title(f"{TASK_LABELS[target]} | n={len(selected)}")
        axis.set_xlabel(f"True current - previous ({TASK_UNITS[target]})")
        axis.set_ylabel(f"Predicted current - previous ({TASK_UNITS[target]})")
        axis.grid(alpha=0.18)
    axes.flat[0].legend(loc="upper left", fontsize=7, frameon=True)
    figure.suptitle(
        f"{model_label}: predicted versus observed change from latest prior value",
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        "Symmetric axes use the pooled 98th percentile to limit extreme-value compression.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_dir / "predicted_vs_observed_change.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _validate_task_record_alignment(input_dir, history_dir, targets):
    if input_dir == history_dir:
        return
    stable_columns = (
        "hospital_id",
        "video_id",
        "source_sample_id",
        "split",
        "raw_value",
    )
    for target in targets:
        prediction_records = pd.read_csv(
            input_dir / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str, "source_sample_id": str},
        ).sort_values("video_id").reset_index(drop=True)
        history_records = pd.read_csv(
            history_dir / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str, "source_sample_id": str},
        ).sort_values("video_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            prediction_records[list(stable_columns)],
            history_records[list(stable_columns)],
            check_dtype=False,
            check_exact=True,
        )


def run(
    input_dir,
    output_dir,
    bootstrap_replicates,
    history_dir=None,
    model_label="Face plus history Head32 regression",
    prediction_architecture="efficientnet_b0",
):
    input_dir = Path(input_dir).resolve()
    history_dir = Path(history_dir or input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    run_index = pd.read_csv(input_dir / "run_index.csv")
    completed = set(run_index.loc[run_index["status"].eq("ok"), "target"].astype(str))
    targets = [target for target in TARGETS if target in completed]
    if not targets:
        raise RuntimeError(f"No completed targets in {input_dir}")
    _validate_task_record_alignment(input_dir, history_dir, targets)

    all_events = []
    audits = []
    summaries = []
    metric_rows = []
    for target_index, target in enumerate(targets):
        audit, eligible = _load_target(
            input_dir, history_dir, target, prediction_architecture
        )
        event = _aggregate_events(eligible)
        evaluable = event.loc[event["direction_evaluable"]].copy()
        metrics = _direction_metrics(evaluable)
        metrics.update(
            _cluster_bootstrap(
                evaluable,
                bootstrap_replicates,
                BOOTSTRAP_SEED + target_index,
            )
        )
        metrics["target"] = target
        metric_rows.append(metrics)
        summaries.append(
            {
                "target": target,
                "test_videos": len(audit),
                "test_videos_with_history": int(audit["history_count"].fillna(0).gt(0).sum()),
                "excluded_no_history": int(
                    audit["exclusion_reason"].eq("no_strictly_prior_same_episode_value").sum()
                ),
                "excluded_ambiguous_latest_value": int(
                    audit["exclusion_reason"].eq(
                        "conflicting_values_at_latest_history_time"
                    ).sum()
                ),
                "eligible_events_with_history": len(event),
                "unchanged_events": int((~event["direction_evaluable"]).sum()),
                "direction_evaluable_events": len(evaluable),
                "direction_evaluable_patients": evaluable["hospital_id"].nunique(),
                "multi_video_events": int(event["video_count"].gt(1).sum()),
            }
        )
        audits.append(audit)
        all_events.append(event)

    events = pd.concat(all_events, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(summaries)
    metrics = metrics[["target", *[column for column in metrics if column != "target"]]]
    events.to_csv(table_dir / "test_event_direction_predictions.csv", index=False)
    audit.to_csv(table_dir / "test_video_eligibility_audit.csv", index=False)
    summary.to_csv(table_dir / "eligibility_summary.csv", index=False)
    metrics.to_csv(table_dir / "direction_metrics.csv", index=False)

    _plot_performance(metrics, figure_dir, model_label)
    _plot_confusions(events, figure_dir, model_label)
    _plot_deltas(events, figure_dir, model_label)
    manifest = {
        "schema_version": 1,
        "analysis": "current lab increase/decrease relative to latest prior lab",
        "model_label": model_label,
        "prediction_source": "saved video-level test predictions; no model inference or retraining",
        "input_dir": str(input_dir),
        "history_dir": str(history_dir),
        "prediction_architecture": prediction_architecture,
        "targets": targets,
        "evaluation_unit": "unique current laboratory event",
        "video_aggregation": "mean predicted current value when multiple videos share one event",
        "history_policy": (
            "latest same-analyte value strictly before current label within the same unique admission"
        ),
        "true_direction": "sign(current raw value - latest prior raw value)",
        "predicted_direction": "sign(predicted current raw value - latest prior raw value)",
        "unchanged_policy": "exclude exact true ties from increase/decrease metrics and report count",
        "ambiguous_latest_policy": "exclude latest timestamps containing conflicting values",
        "primary_metrics": ["balanced_accuracy", "roc_auc", "accuracy"],
        "confidence_intervals": {
            "method": "percentile bootstrap resampling patients with replacement",
            "replicates": bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "level": 0.95,
        },
        "source_fingerprints": {
            "run_index_sha256": _sha256(input_dir / "run_index.csv"),
            "per_target": {
                target: {
                    "predictions_sha256": _sha256(
                        _run_dir(input_dir, target, prediction_architecture)
                        / "video_predictions.csv"
                    ),
                    "task_records_sha256": _sha256(
                        input_dir / "task_records" / f"{target}.csv"
                    ),
                    "history_sha256": _sha256(
                        history_dir / "history_records" / f"{target}.csv"
                    ),
                }
                for target in targets
            },
        },
    }
    (table_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved change-direction analysis to {output_dir}")
    print(metrics[["target", "n", "balanced_accuracy", "roc_auc", "accuracy"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--history-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--model-label", default="Face plus history Head32 regression"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    run(
        args.input_dir,
        args.output_dir,
        args.bootstrap_replicates,
        history_dir=args.history_dir,
        model_label=args.model_label,
    )


if __name__ == "__main__":
    main()
