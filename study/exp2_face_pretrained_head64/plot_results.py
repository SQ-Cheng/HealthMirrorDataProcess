"""Generate figures for aligned 20-frame Head64 binary classification."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
REGRESSION_DIR = (
    EXP_DIR.parent
    / "exp2_face_pretrained_head32_regression"
    / "outputs"
    / "20frame"
)
FIGURE_DIR = OUTPUT_DIR / "figures"

ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
ARCHITECTURE_COLORS = {
    "mobilenet_v3_small": "#2878B5",
    "efficientnet_b0": "#D95F02",
}
TASKS = ("hemoglobin_low", "po2_low", "lactate_high")
COMPARISON_TASKS = ("hemoglobin_low", "po2_low")
TASK_LABELS = {
    "hemoglobin_low": "Hemoglobin low",
    "pco2_low": "pCO2 low",
    "po2_low": "pO2 low",
    "lactate_high": "Lactate high",
}
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F2A541", "test": "#59A14F"}
METHOD_COLORS = {"binary": "#197278", "regression": "#E07A5F"}


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def _validate_complete(output_dir, regression=False, tasks=TASKS):
    run_index = pd.read_csv(output_dir / "run_index.csv")
    keys = ["architecture", "target"]
    if run_index.duplicated(keys).any():
        raise ValueError(f"Duplicate jobs in {output_dir}")
    expected = {
        (architecture, target)
        for architecture in ARCHITECTURES
        for target in tasks
    }
    successful = set(
        run_index.loc[run_index["status"].eq("ok"), keys].itertuples(
            index=False, name=None
        )
    )
    if not expected.issubset(successful):
        raise RuntimeError(f"Incomplete results in {output_dir}: {expected-successful}")
    if not regression and successful != expected:
        raise RuntimeError(f"Unexpected binary jobs in {output_dir}: {successful-expected}")


def _test_class_counts():
    rows = []
    for task in TASKS:
        records = pd.read_csv(
            OUTPUT_DIR / "task_records" / f"{task}.csv",
            dtype={"hospital_id": str, "video_id": str},
        )
        for split in ("train", "val", "test"):
            selected = records.loc[records["split"].eq(split)]
            rows.append(
                {
                    "target": task,
                    "split": split,
                    "negative": int(selected["binary_label"].eq(0).sum()),
                    "positive": int(selected["binary_label"].eq(1).sum()),
                    "videos": int(len(selected)),
                    "positive_rate": float(selected["binary_label"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_dataset_balance(balance):
    x = np.arange(len(TASKS))
    labels = [TASK_LABELS[task] for task in TASKS]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5), squeeze=False)
    for axis, split in zip(axes.flat[:3], ("train", "val", "test")):
        selected = balance.loc[balance["split"].eq(split)].set_index("target").loc[
            list(TASKS)
        ]
        axis.bar(x, selected["negative"], color="#6B7280", label="Negative")
        axis.bar(
            x,
            selected["positive"],
            bottom=selected["negative"],
            color="#E15759",
            label="Positive",
        )
        for position, row in zip(x, selected.itertuples()):
            axis.annotate(
                f"{int(row.positive)}+ / {int(row.negative)}-",
                (position, row.videos),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.set_ylabel("Videos")
        axis.set_title({"train": "Train", "val": "Validation", "test": "Test"}[split])
        axis.grid(axis="y", alpha=0.23)
        axis.legend()

    axis = axes[1, 1]
    width = 0.24
    for offset, split in enumerate(("train", "val", "test")):
        selected = balance.loc[balance["split"].eq(split)].set_index("target").loc[
            list(TASKS)
        ]
        axis.bar(
            x + (offset - 1) * width,
            selected["positive_rate"],
            width=width,
            color=SPLIT_COLORS[split],
            label={"train": "Train", "val": "Validation", "test": "Test"}[split],
        )
    axis.set_ylim(0, 1.04)
    axis.set_xticks(x, labels, rotation=22, ha="right")
    axis.set_ylabel("Positive video fraction")
    axis.set_title("Class prevalence")
    axis.grid(axis="y", alpha=0.23)
    axis.legend()
    figure.suptitle(
        "Aligned patient-level split: binary class balance",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "dataset_balance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _stage_boundary(axis, selected):
    finetune = selected.loc[selected["stage"].eq("finetune"), "global_epoch"]
    if not finetune.empty:
        axis.axvline(
            finetune.min() - 0.5,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
        )


def plot_training_curves(history, metrics):
    test = metrics.loc[metrics["split"].eq("test")].set_index(
        ["architecture", "target"]
    )
    figure, axes = plt.subplots(
        len(TASKS),
        2,
        figsize=(15, max(8, 4.0 * len(TASKS))),
        squeeze=False,
    )
    for row, task in enumerate(TASKS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            selected = history.loc[
                history["architecture"].eq(architecture)
                & history["target"].eq(task)
            ].sort_values("global_epoch")
            epochs = selected["global_epoch"]
            axis.plot(
                epochs,
                selected["train_eval_loss"],
                color="#4C78A8",
                label="Train BCE",
            )
            axis.plot(
                epochs,
                selected["val_loss"],
                color="#F2A541",
                label="Validation BCE",
            )
            _stage_boundary(axis, selected)
            axis.set_xlabel("Global epoch")
            axis.set_ylabel("Weighted BCE loss")
            axis.grid(axis="y", alpha=0.22)
            metric_axis = axis.twinx()
            metric_axis.plot(
                epochs,
                selected["train_roc_auc"],
                color="#197278",
                linestyle="--",
                alpha=0.72,
                label="Train AUC",
            )
            metric_axis.plot(
                epochs,
                selected["val_roc_auc"],
                color="#E15759",
                linestyle="--",
                alpha=0.82,
                label="Validation AUC",
            )
            metric_axis.axhline(0.5, color="#777777", linestyle=":", linewidth=0.7)
            metric_axis.set_ylim(0, 1.04)
            metric_axis.set_ylabel("Video-level ROC-AUC")
            row_metrics = test.loc[(architecture, task)]
            axis.set_title(
                f"{TASK_LABELS[task]} | {ARCHITECTURE_LABELS[architecture]}\n"
                f"selected={row_metrics['selected_stage']}"
            )
            handles_a, labels_a = axis.get_legend_handles_labels()
            handles_b, labels_b = metric_axis.get_legend_handles_labels()
            axis.legend(handles_a + handles_b, labels_a + labels_b, ncol=2)
    figure.suptitle(
        "Aligned 20-frame Head64 binary classification: training history\n"
        "Vertical dotted line marks backbone unfreezing",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_test_metrics(metrics, balance):
    test = metrics.loc[metrics["split"].eq("test")].set_index(
        ["architecture", "target"]
    )
    test_balance = balance.loc[balance["split"].eq("test")].set_index("target")
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.36
    labels = [
        f"{TASK_LABELS[task]}\n"
        f"{int(test_balance.loc[task, 'positive'])}+/"
        f"{int(test_balance.loc[task, 'negative'])}-"
        for task in TASKS
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5), squeeze=False)
    specifications = (
        ("roc_auc", "ROC-AUC", 0.5),
        ("balanced_accuracy", "Balanced accuracy", 0.5),
        ("average_precision", "Average precision", None),
        ("f1", "F1 score", None),
    )
    for axis, (metric, ylabel, baseline) in zip(axes.flat, specifications):
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            selected = test.loc[(architecture, list(TASKS)), metric].to_numpy()
            axis.bar(
                x + (architecture_index - 0.5) * width,
                selected,
                width=width,
                color=ARCHITECTURE_COLORS[architecture],
                label=ARCHITECTURE_LABELS[architecture],
            )
        if baseline is not None:
            axis.axhline(baseline, color="#555555", linestyle="--", linewidth=0.9)
        if metric == "average_precision":
            axis.scatter(
                x,
                test_balance.loc[list(TASKS), "positive_rate"],
                marker="D",
                facecolors="white",
                edgecolors="#222222",
                s=42,
                zorder=5,
                label="Positive-rate baseline",
            )
        axis.set_ylim(0, 1.04)
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.23)
        axis.legend()
    figure.suptitle(
        "Aligned 20-frame Head64 binary classification: video-level test metrics",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_binary_test(architecture, task):
    predictions = pd.read_csv(
        OUTPUT_DIR / "runs" / architecture / task / "video_predictions.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    return predictions.loc[predictions["split"].eq("test")].copy()


def plot_roc_curves(metrics):
    test = metrics.loc[metrics["split"].eq("test")].set_index(
        ["architecture", "target"]
    )
    figure, axes = plt.subplots(1, len(TASKS), figsize=(13, 5.5), squeeze=False)
    for axis, task in zip(axes.flat, TASKS):
        positive = negative = 0
        for architecture in ARCHITECTURES:
            predictions = _load_binary_test(architecture, task)
            labels = predictions["y_true"].to_numpy(np.uint8)
            scores = predictions["score"].to_numpy(np.float64)
            false_positive, true_positive, _ = roc_curve(labels, scores)
            row = test.loc[(architecture, task)]
            axis.plot(
                false_positive,
                true_positive,
                linewidth=1.8,
                color=ARCHITECTURE_COLORS[architecture],
                label=f"{ARCHITECTURE_LABELS[architecture]} (AUC={row.roc_auc:.3f})",
            )
            positive = int(labels.sum())
            negative = int(len(labels) - positive)
        axis.plot((0, 1), (0, 1), color="#666666", linestyle="--", linewidth=0.9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_title(f"{TASK_LABELS[task]} | {positive}+/{negative}-")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Video-level test ROC curves", fontsize=15)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "test_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices():
    figure, axes = plt.subplots(
        len(TASKS),
        2,
        figsize=(11, max(8, 4.0 * len(TASKS))),
        squeeze=False,
    )
    for row, task in enumerate(TASKS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            predictions = _load_binary_test(architecture, task)
            labels = predictions["y_true"].to_numpy(np.uint8)
            decisions = (
                predictions["score"].to_numpy(np.float64)
                >= predictions["threshold"].iloc[0]
            ).astype(np.uint8)
            matrix = confusion_matrix(labels, decisions, labels=[0, 1])
            denominators = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            normalized = matrix / denominators
            axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for y in range(2):
                for x in range(2):
                    axis.text(
                        x,
                        y,
                        f"{matrix[y, x]}\n{normalized[y, x]:.1%}",
                        ha="center",
                        va="center",
                        color="white" if normalized[y, x] > 0.52 else "#222222",
                        fontsize=10,
                    )
            axis.set_xticks((0, 1), ("Predicted -", "Predicted +"))
            axis.set_yticks((0, 1), ("True -", "True +"))
            axis.set_title(
                f"{TASK_LABELS[task]} | {ARCHITECTURE_LABELS[architecture]}\n"
                f"threshold={predictions['threshold'].iloc[0]:.4g}"
            )
    figure.suptitle(
        "Video-level test confusion matrices",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975), h_pad=3.2)
    figure.savefig(
        FIGURE_DIR / "test_confusion_matrices.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _logit(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(values / (1 - values))


def plot_score_distributions():
    rng = np.random.default_rng(20260729)
    figure, axes = plt.subplots(
        len(TASKS),
        2,
        figsize=(12, max(8, 4.0 * len(TASKS))),
        squeeze=False,
    )
    for row, task in enumerate(TASKS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            predictions = _load_binary_test(architecture, task)
            labels = predictions["y_true"].to_numpy(np.uint8)
            logits = _logit(predictions["score"])
            groups = [logits[labels == value] for value in (0, 1)]
            box = axis.boxplot(
                groups,
                positions=(0, 1),
                widths=0.45,
                patch_artist=True,
                showfliers=False,
            )
            for patch, color in zip(box["boxes"], ("#6B7280", "#E15759")):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            for value, group, color in zip((0, 1), groups, ("#6B7280", "#E15759")):
                jitter = rng.uniform(-0.13, 0.13, len(group))
                axis.scatter(
                    value + jitter,
                    group,
                    s=24,
                    alpha=0.72,
                    color=color,
                    edgecolors="none",
                )
            threshold = float(predictions["threshold"].iloc[0])
            axis.axhline(
                _logit([threshold])[0],
                color="#197278",
                linestyle="--",
                linewidth=1.1,
                label=f"Validation threshold={threshold:.3g}",
            )
            axis.set_xticks((0, 1), ("Negative", "Positive"))
            axis.set_ylabel("Predicted logit score")
            axis.set_title(f"{TASK_LABELS[task]} | {ARCHITECTURE_LABELS[architecture]}")
            axis.grid(axis="y", alpha=0.22)
            axis.legend()
    figure.suptitle(
        "Video-level test score distributions and selected thresholds",
        fontsize=15,
        y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.982), h_pad=2.2)
    figure.savefig(
        FIGURE_DIR / "test_score_distributions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def binary_regression_comparison(binary_metrics, regression_metrics):
    rows = []
    for architecture in ARCHITECTURES:
        for task in COMPARISON_TASKS:
            binary_predictions = _load_binary_test(architecture, task)
            regression_predictions = pd.read_csv(
                REGRESSION_DIR
                / "runs"
                / architecture
                / task
                / "video_predictions.csv",
                dtype={"hospital_id": str, "video_id": str},
            )
            regression_predictions = regression_predictions.loc[
                regression_predictions["split"].eq("test")
            ]
            paired = binary_predictions.merge(
                regression_predictions,
                on=["hospital_id", "video_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("_binary", "_regression"),
            )
            if (
                len(paired) != len(binary_predictions)
                or len(paired) != len(regression_predictions)
            ):
                raise ValueError(f"Unpaired test set for {architecture}/{task}")
            if not np.array_equal(
                paired["y_true_binary"].to_numpy(np.uint8),
                paired["binary_label"].to_numpy(np.uint8),
            ):
                raise ValueError(f"Label mismatch for {architecture}/{task}")
            binary_row = binary_metrics.loc[
                binary_metrics["architecture"].eq(architecture)
                & binary_metrics["target"].eq(task)
                & binary_metrics["split"].eq("test")
            ].iloc[0]
            regression_row = regression_metrics.loc[
                regression_metrics["architecture"].eq(architecture)
                & regression_metrics["target"].eq(task)
                & regression_metrics["split"].eq("test")
            ].iloc[0]
            for method, row in (
                ("binary", binary_row),
                ("regression", regression_row),
            ):
                rows.append(
                    {
                        "architecture": architecture,
                        "target": task,
                        "method": method,
                        "n_test_videos": int(len(paired)),
                        "positive_test_videos": int(
                            paired["y_true_binary"].sum()
                        ),
                        "roc_auc": float(
                            row["roc_auc"]
                            if method == "binary"
                            else row["sign_roc_auc"]
                        ),
                        "balanced_accuracy": float(
                            row["balanced_accuracy"]
                            if method == "binary"
                            else row["sign_balanced_accuracy"]
                        ),
                    }
                )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        FIGURE_DIR / "binary_vs_regression_test_metrics.csv",
        index=False,
    )
    return comparison


def plot_binary_regression_comparison(comparison):
    x = np.arange(len(COMPARISON_TASKS), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5), squeeze=False)
    for row, architecture in enumerate(ARCHITECTURES):
        selected_architecture = comparison.loc[
            comparison["architecture"].eq(architecture)
        ]
        counts = (
            selected_architecture.loc[selected_architecture["method"].eq("binary")]
            .set_index("target")
            .loc[list(COMPARISON_TASKS)]
        )
        labels = [
            f"{TASK_LABELS[task]}\n"
            f"{int(counts.loc[task, 'positive_test_videos'])}+/"
            f"{int(counts.loc[task, 'n_test_videos'] - counts.loc[task, 'positive_test_videos'])}-"
            for task in COMPARISON_TASKS
        ]
        for column, (metric, ylabel) in enumerate(
            (("roc_auc", "ROC-AUC"), ("balanced_accuracy", "Balanced accuracy"))
        ):
            axis = axes[row, column]
            for method_index, method in enumerate(("binary", "regression")):
                selected = (
                    selected_architecture.loc[
                        selected_architecture["method"].eq(method)
                    ]
                    .set_index("target")
                    .loc[list(COMPARISON_TASKS)]
                )
                axis.bar(
                    x + (method_index - 0.5) * width,
                    selected[metric],
                    width=width,
                    color=METHOD_COLORS[method],
                    label=(
                        "Direct binary classifier"
                        if method == "binary"
                        else "Regression score sign"
                    ),
                )
            axis.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9)
            axis.set_ylim(0, 1.04)
            axis.set_xticks(x, labels, rotation=22, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(ARCHITECTURE_LABELS[architecture])
            axis.grid(axis="y", alpha=0.23)
            axis.legend()
    figure.suptitle(
        "Direct binary classification vs abnormal-score regression\n"
        "Strict comparison: identical patient split, test videos, and binary labels",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "binary_vs_regression_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(output_dir=None, regression_dir=None):
    global OUTPUT_DIR, REGRESSION_DIR, FIGURE_DIR
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir).resolve()
    if regression_dir is not None:
        REGRESSION_DIR = Path(regression_dir).resolve()
    FIGURE_DIR = OUTPUT_DIR / "figures"
    _style()
    _validate_complete(OUTPUT_DIR)
    _validate_complete(
        REGRESSION_DIR,
        regression=True,
        tasks=COMPARISON_TASKS,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(OUTPUT_DIR / "history_all.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "metrics_all.csv")
    regression_metrics = pd.read_csv(REGRESSION_DIR / "metrics_all.csv")
    balance = _test_class_counts()
    balance.to_csv(FIGURE_DIR / "dataset_balance.csv", index=False)
    plot_dataset_balance(balance)
    plot_training_curves(history, metrics)
    plot_test_metrics(metrics, balance)
    plot_roc_curves(metrics)
    plot_confusion_matrices()
    plot_score_distributions()
    plot_binary_regression_comparison(
        binary_regression_comparison(metrics, regression_metrics)
    )
    print(f"Saved 7 figures and 2 tables to {FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--regression-dir", default=None)
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.regression_dir)
