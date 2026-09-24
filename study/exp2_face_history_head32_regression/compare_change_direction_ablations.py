"""Paired comparison of direction prediction across three regression pathways."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .analyze_change_direction import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from .config import TARGETS
from .plot_results import TASK_LABELS


EXPERIMENT_DIR = Path(__file__).resolve().parent
STUDY_DIR = EXPERIMENT_DIR.parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs/change_direction_ablation_comparison"
MODEL_SOURCES = {
    "face_history": {
        "label": "Face + history",
        "color": "#3A7D44",
        "root": EXPERIMENT_DIR / "outputs/20frame/change_direction_analysis",
    },
    "face_only": {
        "label": "Face only",
        "color": "#2F6B8A",
        "root": STUDY_DIR
        / "exp2_face_pretrained_head32_regression/outputs/20frame/change_direction_analysis",
    },
    "history_only": {
        "label": "History only",
        "color": "#D18F2F",
        "root": STUDY_DIR
        / "exp2_history_only_head32_regression/outputs/change_direction_analysis",
    },
}

COMPARISONS = (
    ("face_history", "face_only", "Face + history minus face only", "#3A7D44"),
    ("face_history", "history_only", "Face + history minus history only", "#3F8E8E"),
    ("face_only", "history_only", "Face only minus history only", "#7A7A7A"),
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sources():
    events = {}
    metrics = []
    for model, specification in MODEL_SOURCES.items():
        table_dir = specification["root"] / "tables"
        event = pd.read_csv(
            table_dir / "test_event_direction_predictions.csv",
            dtype={"hospital_id": str, "source_sample_id": str},
        )
        event = event.loc[event["direction_evaluable"]].copy()
        event["model"] = model
        events[model] = event
        metric = pd.read_csv(table_dir / "direction_metrics.csv")
        metric["model"] = model
        metric["model_label"] = specification["label"]
        metrics.append(metric)
    return events, pd.concat(metrics, ignore_index=True)


def _align_events(events):
    key = ["target", "hospital_id", "source_sample_id"]
    reference = events["face_history"].sort_values(key).reset_index(drop=True)
    aligned = {}
    audit = []
    for model, frame in events.items():
        current = frame.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            reference[key], current[key], check_dtype=False, check_exact=True
        )
        for column in ("true_increase", "true_delta", "previous_value"):
            if not np.allclose(
                reference[column].to_numpy(np.float64),
                current[column].to_numpy(np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(f"{model} differs in aligned {column}")
        aligned[model] = current
        audit.append(
            {
                "model": model,
                "model_label": MODEL_SOURCES[model]["label"],
                "events": len(current),
                "patients": current["hospital_id"].nunique(),
                "alignment": "exact",
            }
        )
    return aligned, pd.DataFrame(audit)


def _metric_values(true, predicted, score):
    positive = true == 1
    negative = ~positive
    balanced_accuracy = 0.5 * (
        predicted[positive].mean() + (predicted[negative] == 0).mean()
    )
    return {
        "balanced_accuracy": float(balanced_accuracy),
        "roc_auc": float(roc_auc_score(true, score)),
    }


def _paired_bootstrap(aligned, replicates=BOOTSTRAP_REPLICATES):
    rows = []
    for target_index, target in enumerate(TARGETS):
        target_frames = {
            model: frame.loc[frame["target"].eq(target)].reset_index(drop=True)
            for model, frame in aligned.items()
        }
        reference = target_frames["face_history"]
        true = reference["true_increase"].to_numpy(np.uint8)
        patients = reference["hospital_id"].astype(str).to_numpy()
        unique_patients = np.unique(patients)
        patient_positions = {
            patient: np.flatnonzero(patients == patient) for patient in unique_patients
        }
        predictions = {
            model: frame["predicted_increase"].to_numpy(np.uint8)
            for model, frame in target_frames.items()
        }
        scores = {
            model: frame["predicted_delta"].to_numpy(np.float64)
            for model, frame in target_frames.items()
        }
        point = {
            model: _metric_values(true, predictions[model], scores[model])
            for model in MODEL_SOURCES
        }
        samples = {
            (first, second, metric): []
            for first, second, _, _ in COMPARISONS
            for metric in ("balanced_accuracy", "roc_auc")
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED + 100 + target_index)
        for _ in range(replicates):
            selected_patients = rng.choice(
                unique_patients, size=len(unique_patients), replace=True
            )
            positions = np.concatenate(
                [patient_positions[patient] for patient in selected_patients]
            )
            bootstrap_true = true[positions]
            if len(np.unique(bootstrap_true)) != 2:
                continue
            bootstrap_metrics = {
                model: _metric_values(
                    bootstrap_true,
                    predictions[model][positions],
                    scores[model][positions],
                )
                for model in MODEL_SOURCES
            }
            for first, second, _, _ in COMPARISONS:
                for metric in ("balanced_accuracy", "roc_auc"):
                    samples[(first, second, metric)].append(
                        bootstrap_metrics[first][metric]
                        - bootstrap_metrics[second][metric]
                    )
        for first, second, label, _ in COMPARISONS:
            for metric in ("balanced_accuracy", "roc_auc"):
                values = samples[(first, second, metric)]
                rows.append(
                    {
                        "target": target,
                        "metric": metric,
                        "first_model": first,
                        "second_model": second,
                        "comparison": label,
                        "estimate": point[first][metric] - point[second][metric],
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "bootstrap_valid": len(values),
                        "patients": len(unique_patients),
                        "events": len(reference),
                    }
                )
    return pd.DataFrame(rows)


def _plot_model_metrics(metrics, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    targets = list(TARGETS)
    y = np.arange(len(targets), dtype=float)
    offsets = {"face_history": -0.20, "face_only": 0.0, "history_only": 0.20}
    for axis, metric, title in zip(
        axes,
        ("balanced_accuracy", "roc_auc"),
        ("Balanced accuracy", "ROC AUC"),
    ):
        for model, specification in MODEL_SOURCES.items():
            selected = metrics.loc[metrics["model"].eq(model)].set_index("target")
            values = selected.loc[targets, metric].to_numpy(float)
            low = selected.loc[targets, f"{metric}_ci_low"].to_numpy(float)
            high = selected.loc[targets, f"{metric}_ci_high"].to_numpy(float)
            axis.errorbar(
                values,
                y + offsets[model],
                xerr=np.vstack((values - low, high - values)),
                fmt="o",
                ms=6,
                capsize=2.5,
                linewidth=1.2,
                color=specification["color"],
                label=specification["label"],
            )
        axis.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Estimate with patient-clustered 95% CI", xlim=(0.3, 1.0))
        axis.grid(axis="x", alpha=0.25)
    counts = (
        metrics.loc[metrics["model"].eq("face_history")]
        .set_index("target")
        .loc[targets, "n"]
        .astype(int)
    )
    axes[0].set_yticks(
        y, [f"{TASK_LABELS[target]}  (n={counts[target]})" for target in targets]
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Direction prediction across matched regression pathways", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(
        output_dir / "three_model_direction_performance.png",
        dpi=210,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_paired_differences(differences, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    targets = list(TARGETS)
    y = np.arange(len(targets), dtype=float)
    offsets = (-0.20, 0.0, 0.20)
    for axis, metric, title in zip(
        axes,
        ("balanced_accuracy", "roc_auc"),
        ("Paired difference in balanced accuracy", "Paired difference in ROC AUC"),
    ):
        metric_rows = differences.loc[differences["metric"].eq(metric)]
        for offset, (first, second, label, color) in zip(offsets, COMPARISONS):
            selected = metric_rows.loc[
                metric_rows["first_model"].eq(first)
                & metric_rows["second_model"].eq(second)
            ].set_index("target").loc[targets]
            values = selected["estimate"].to_numpy(float)
            low = selected["ci_low"].to_numpy(float)
            high = selected["ci_high"].to_numpy(float)
            axis.errorbar(
                values,
                y + offset,
                xerr=np.vstack((values - low, high - values)),
                fmt="o",
                ms=6,
                capsize=2.5,
                linewidth=1.2,
                color=color,
                label=label,
            )
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("First model minus second model (paired patient bootstrap 95% CI)")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [TASK_LABELS[target] for target in targets])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("Paired pathway differences on identical test events", y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(
        output_dir / "paired_model_differences.png", dpi=210, bbox_inches="tight"
    )
    plt.close(figure)


def main():
    figure_dir = OUTPUT_DIR / "figures"
    table_dir = OUTPUT_DIR / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    events, metrics = _load_sources()
    aligned, alignment = _align_events(events)
    differences = _paired_bootstrap(aligned)
    metrics.to_csv(table_dir / "three_model_direction_metrics.csv", index=False)
    differences.to_csv(table_dir / "paired_model_differences.csv", index=False)
    alignment.to_csv(table_dir / "alignment_audit.csv", index=False)
    _plot_model_metrics(metrics, figure_dir)
    _plot_paired_differences(differences, figure_dir)
    manifest = {
        "schema_version": 1,
        "analysis": "paired comparison of change-direction prediction pathways",
        "models": {
            model: {
                "label": specification["label"],
                "source": str(specification["root"]),
                "event_predictions_sha256": _sha256(
                    specification["root"]
                    / "tables/test_event_direction_predictions.csv"
                ),
                "metrics_sha256": _sha256(
                    specification["root"] / "tables/direction_metrics.csv"
                ),
            }
            for model, specification in MODEL_SOURCES.items()
        },
        "alignment": "exact target, hospital_id, source_sample_id, true direction, true delta, and previous value",
        "paired_confidence_intervals": {
            "unit": "patient",
            "method": "percentile cluster bootstrap with shared resamples across models",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_base": BOOTSTRAP_SEED + 100,
            "level": 0.95,
        },
    }
    (table_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    macro = (
        metrics.groupby(["model", "model_label"])[
            ["balanced_accuracy", "roc_auc", "accuracy"]
        ]
        .mean()
        .reset_index()
    )
    macro.to_csv(table_dir / "macro_average_metrics.csv", index=False)
    print(f"Saved paired comparison to {OUTPUT_DIR}")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
