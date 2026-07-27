"""Generate video-level result figures for abnormal-score regression."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
EXPERIMENT_LABEL = "Distribution-balanced abnormal-score regression"
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
                selected["train_eval_loss"],
                color="#2878B5",
                linewidth=1.3,
                label="Train evaluation loss",
            )
            axis.plot(
                epochs,
                selected["val_loss"],
                color="#E15759",
                linewidth=1.3,
                label="Validation loss",
            )
            axis.set_ylabel("SmoothL1 loss")
            score_axis = axis.twinx()
            score_axis.plot(
                epochs,
                selected["val_mae"],
                color="#F2A541",
                linewidth=1.1,
                label="Validation MAE",
            )
            score_axis.plot(
                epochs,
                selected["val_pearson_r"],
                color="#59A14F",
                linewidth=1.1,
                label="Validation Pearson r",
            )
            score_axis.set_ylabel("Video-level metric")
            stage_change = selected.loc[selected["stage"].eq("finetune"), "global_epoch"]
            if not stage_change.empty:
                axis.axvline(
                    stage_change.min() - 0.5,
                    color="#666666",
                    linestyle=":",
                    linewidth=1,
                )
            axis.axhline(0, color="#888888", linewidth=0.6)
            axis.grid(axis="y", alpha=0.22)
            axis.set_xlabel("Epoch")
            axis.set_title(
                f"{TASK_LABELS[target]} | {ARCHITECTURE_LABELS[architecture]}"
            )
            handles_a, labels_a = axis.get_legend_handles_labels()
            handles_b, labels_b = score_axis.get_legend_handles_labels()
            axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    figure.suptitle(
        f"{EXPERIMENT_LABEL}: training history",
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
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    specifications = (
        (axes[0, 0], ("mae", "rmse"), ("MAE", "RMSE"), "Prediction error", None),
        (
            axes[0, 1],
            ("pearson_r", "spearman_r"),
            ("Pearson r", "Spearman r"),
            "Correlation",
            (-1.05, 1.05),
        ),
        (axes[1, 0], ("r2",), ("R2",), "Coefficient of determination", None),
        (
            axes[1, 1],
            ("sign_roc_auc", "sign_balanced_accuracy"),
            ("Sign ROC-AUC", "Sign bACC"),
            "Zero-boundary classification",
            (0, 1.04),
        ),
    )
    for axis, columns, labels, ylabel, ylim in specifications:
        subwidth = width / len(columns)
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            selected = (
                test.loc[test["architecture"].eq(architecture)]
                .set_index("target")
                .loc[list(TASKS)]
            )
            center = x + (architecture_index - 0.5) * width
            for metric_index, (column, label) in enumerate(zip(columns, labels)):
                position = center + (
                    metric_index - (len(columns) - 1) / 2
                ) * subwidth
                axis.bar(
                    position,
                    selected[column],
                    width=subwidth,
                    color=COLORS[architecture],
                    alpha=0.92 if metric_index == 0 else 0.52,
                    hatch=None if metric_index == 0 else "//",
                    label=f"{ARCHITECTURE_LABELS[architecture]} | {label}",
                )
        if "sign_roc_auc" in columns:
            axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        else:
            axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            x, [TASK_LABELS[target] for target in TASKS], rotation=22, ha="right"
        )
        axis.grid(axis="y", alpha=0.24)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle(
        f"{EXPERIMENT_LABEL}: video-level test performance", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_split_generalization(metrics):
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.24
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), squeeze=False)
    for row, architecture in enumerate(ARCHITECTURES):
        selected = metrics.loc[metrics["architecture"].eq(architecture)]
        for column, (metric, ylabel) in enumerate(
            (("mae", "Video-level MAE"), ("pearson_r", "Video-level Pearson r"))
        ):
            axis = axes[row, column]
            for offset, split in enumerate(("train", "val", "test")):
                values = (
                    selected.loc[selected["split"].eq(split)]
                    .set_index("target")
                    .loc[list(TASKS), metric]
                )
                axis.bar(
                    x + (offset - 1) * width,
                    values,
                    width=width,
                    color=SPLIT_COLORS[split],
                    alpha=0.88,
                    label={
                        "train": "Train",
                        "val": "Validation",
                        "test": "Test",
                    }[split],
                )
            axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
            axis.set_xticks(
                x,
                [TASK_LABELS[target] for target in TASKS],
                rotation=24,
                ha="right",
            )
            axis.set_ylabel(ylabel)
            axis.set_title(ARCHITECTURE_LABELS[architecture])
            axis.grid(axis="y", alpha=0.24)
            axis.legend()
    figure.suptitle(f"{EXPERIMENT_LABEL}: split generalization", fontsize=15)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "split_generalization.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_predictions(metrics):
    test_metrics = metrics.loc[metrics["split"].eq("test")].set_index(
        ["architecture", "target"]
    )
    figure, axes = plt.subplots(5, 2, figsize=(13, 22), squeeze=False)
    for row, target in enumerate(TASKS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            path = (
                OUTPUT_DIR
                / "runs"
                / architecture
                / target
                / "video_predictions.csv"
            )
            predictions = pd.read_csv(path)
            predictions = predictions.loc[predictions["split"].eq("test")]
            y_true = predictions["y_true"].to_numpy(dtype=np.float64)
            y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
            normal = y_true < 0
            axis.scatter(
                y_true[normal],
                y_pred[normal],
                s=22,
                alpha=0.67,
                color="#4C78A8",
                edgecolors="none",
                label="Normal side",
            )
            axis.scatter(
                y_true[~normal],
                y_pred[~normal],
                s=22,
                alpha=0.72,
                color="#E15759",
                edgecolors="none",
                label="Abnormal/boundary side",
            )
            lower = float(min(y_true.min(), y_pred.min()))
            upper = float(max(y_true.max(), y_pred.max()))
            padding = max((upper - lower) * 0.06, 0.05)
            limits = (lower - padding, upper + padding)
            axis.plot(limits, limits, color="#333333", linestyle="--", linewidth=1)
            axis.axhline(0, color="#888888", linewidth=0.7)
            axis.axvline(0, color="#888888", linewidth=0.7)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("True abnormal score")
            axis.set_ylabel("Predicted abnormal score")
            row_metrics = test_metrics.loc[(architecture, target)]
            axis.set_title(
                f"{TASK_LABELS[target]} | {ARCHITECTURE_LABELS[architecture]}\n"
                f"n={int(row_metrics['n'])}, MAE={row_metrics['mae']:.3f}, "
                f"r={row_metrics['pearson_r']:.3f}"
            )
            axis.grid(alpha=0.18)
            axis.legend(loc="best")
    figure.suptitle(
        f"{EXPERIMENT_LABEL}: video-level test predictions",
        fontsize=15,
        y=1.002,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "test_predicted_vs_true.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def main(output_dir=None):
    global OUTPUT_DIR, FIGURE_DIR, EXPERIMENT_LABEL
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir).resolve()
        FIGURE_DIR = OUTPUT_DIR / "figures"
    _style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(OUTPUT_DIR / "history_all.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "metrics_all.csv")
    plot_training_curves(history)
    plot_test_metrics(metrics)
    plot_split_generalization(metrics)
    plot_predictions(metrics)
    print(f"Saved 4 regression figures to {FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args()
    main(arguments.output_dir)
