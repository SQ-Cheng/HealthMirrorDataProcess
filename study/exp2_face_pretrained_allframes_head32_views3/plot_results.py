"""Generate video-level result figures for the three-view classification run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
TASKS = (
    "hemoglobin_low",
    "pco2_low",
    "po2_low",
    "high_blood_pressure",
    "lactate_high",
)
TASK_LABELS = {
    "hemoglobin_low": "Hemoglobin low",
    "pco2_low": "pCO2 low",
    "po2_low": "pO2 low",
    "high_blood_pressure": "High blood pressure",
    "lactate_high": "Lactate high",
}
COLORS = {
    "mobilenet_v3_small": "#2878B5",
    "efficientnet_b0": "#D95F02",
}
TASK_COLORS = ("#2878B5", "#59A14F", "#E15759", "#B07AA1", "#F28E2B")
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F2A541", "test": "#59A14F"}


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_training_curves(history):
    figure, axes = plt.subplots(5, 2, figsize=(15, 19), squeeze=False)
    for row, target in enumerate(TASKS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            selected = history[
                history["architecture"].eq(architecture)
                & history["target"].eq(target)
            ].sort_values("global_epoch")
            epochs = selected["global_epoch"].to_numpy()
            axis.plot(
                epochs,
                np.maximum(selected["train_loss"], 1e-5),
                color="#2878B5",
                linewidth=1.3,
                label="Train loss",
            )
            axis.plot(
                epochs,
                np.maximum(selected["val_loss"], 1e-5),
                color="#E15759",
                linewidth=1.3,
                label="Validation loss",
            )
            axis.set_yscale("log")
            axis.set_ylabel("BCE loss (log scale)")
            score_axis = axis.twinx()
            score_axis.plot(
                epochs,
                selected["val_roc_auc"],
                color="#59A14F",
                linewidth=1.1,
                label="Validation ROC-AUC",
            )
            score_axis.plot(
                epochs,
                selected["val_bacc"],
                color="#B07AA1",
                linewidth=1.1,
                label="Validation bACC",
            )
            score_axis.set_ylim(-0.02, 1.02)
            score_axis.set_ylabel("Video-level score")
            stage_change = selected.loc[selected["stage"].eq("finetune"), "global_epoch"]
            if not stage_change.empty:
                axis.axvline(
                    stage_change.min() - 0.5,
                    color="#666666",
                    linestyle=":",
                    linewidth=1,
                )
            axis.grid(axis="y", alpha=0.22)
            axis.set_xlabel("Epoch")
            axis.set_title(
                f"{TASK_LABELS[target]} | {ARCHITECTURE_LABELS[architecture]}"
            )
            handles_a, labels_a = axis.get_legend_handles_labels()
            handles_b, labels_b = score_axis.get_legend_handles_labels()
            axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    figure.suptitle(
        "Three-view all-frame classification: training history",
        fontsize=15,
        y=1.002,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "training_curves.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_test_metrics(metrics):
    test = metrics.loc[metrics["split"].eq("test")]
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    specifications = (
        ("roc_auc", "ROC-AUC", True),
        ("balanced_accuracy", "Balanced accuracy", True),
        ("average_precision", "Average precision", False),
    )
    for axis, (column, ylabel, chance_line) in zip(axes, specifications):
        for offset, architecture in enumerate(ARCHITECTURES):
            selected = (
                test.loc[test["architecture"].eq(architecture)]
                .set_index("target")
                .loc[list(TASKS)]
            )
            axis.bar(
                x + (offset - 0.5) * width,
                selected[column],
                width=width,
                color=COLORS[architecture],
                alpha=0.88,
                label=ARCHITECTURE_LABELS[architecture],
            )
        if chance_line:
            axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1.04)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.24)
        axis.legend(ncol=2)
    axes[-1].set_xticks(
        x, [TASK_LABELS[target] for target in TASKS], rotation=15, ha="right"
    )
    figure.suptitle("Three-view all-frame classification: test performance", fontsize=15)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_split_auc(metrics):
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    for axis, architecture in zip(axes, ARCHITECTURES):
        selected = metrics.loc[metrics["architecture"].eq(architecture)]
        for offset, split in enumerate(("train", "val", "test")):
            values = (
                selected.loc[selected["split"].eq(split)]
                .set_index("target")
                .loc[list(TASKS), "roc_auc"]
            )
            axis.bar(
                x + (offset - 1) * width,
                values,
                width=width,
                color=SPLIT_COLORS[split],
                alpha=0.88,
                label={"train": "Train", "val": "Validation", "test": "Test"}[split],
            )
        axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1.04)
        axis.set_xticks(
            x, [TASK_LABELS[target] for target in TASKS], rotation=25, ha="right"
        )
        axis.set_title(ARCHITECTURE_LABELS[architecture])
        axis.grid(axis="y", alpha=0.24)
        axis.legend()
    axes[0].set_ylabel("Video-level ROC-AUC")
    figure.suptitle("Three-view classification: split generalization", fontsize=15)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "split_auc.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_test_roc_pr(metrics):
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False)
    test_metrics = metrics.loc[metrics["split"].eq("test")].set_index(
        ["architecture", "target"]
    )
    for row, architecture in enumerate(ARCHITECTURES):
        roc_axis, pr_axis = axes[row]
        for target, color in zip(TASKS, TASK_COLORS):
            path = (
                OUTPUT_DIR
                / "runs"
                / architecture
                / target
                / "video_predictions.csv"
            )
            predictions = pd.read_csv(path)
            predictions = predictions.loc[predictions["split"].eq("test")]
            y_true = predictions["y_true"].to_numpy(dtype=np.uint8)
            score = predictions["score"].to_numpy(dtype=np.float64)
            fpr, tpr, _ = roc_curve(y_true, score)
            precision, recall, _ = precision_recall_curve(y_true, score)
            row_metrics = test_metrics.loc[(architecture, target)]
            roc_axis.plot(
                fpr,
                tpr,
                color=color,
                linewidth=1.5,
                label=f"{TASK_LABELS[target]} ({row_metrics.roc_auc:.3f})",
            )
            pr_axis.plot(
                recall,
                precision,
                color=color,
                linewidth=1.5,
                label=(
                    f"{TASK_LABELS[target]} "
                    f"({row_metrics.average_precision:.3f})"
                ),
            )
        roc_axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
        roc_axis.set(xlabel="False-positive rate", ylabel="True-positive rate")
        pr_axis.set(xlabel="Recall", ylabel="Precision")
        roc_axis.set_title(f"{ARCHITECTURE_LABELS[architecture]} | ROC")
        pr_axis.set_title(f"{ARCHITECTURE_LABELS[architecture]} | precision-recall")
        for axis in (roc_axis, pr_axis):
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.22)
            axis.legend(loc="best")
    figure.suptitle("Three-view classification: video-level test curves", fontsize=15)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "test_roc_pr_curves.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def main():
    _style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(OUTPUT_DIR / "history_all.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "metrics_all.csv")
    plot_training_curves(history)
    plot_test_metrics(metrics)
    plot_split_auc(metrics)
    plot_test_roc_pr(metrics)
    print(f"Saved 4 classification figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
