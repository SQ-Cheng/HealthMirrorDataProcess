"""Per-model figures and paired architecture comparisons on identical videos."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape, target_grid_figsize
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from study.exp2_face_pretrained_head32_classification.plot_confusion_matrices import plot_confusion_matrices

from . import config


LABELS = {"efficientnet_b0": "Pretrained EfficientNet-B0", "color_histogram_mlp": "Color histogram + MLP",
          "color_statistics_mlp": "Color statistics + MLP", "small_cnn": "Small CNN (97k)"}
COLORS = ("#64737C", "#2878B5", "#CB6547", "#278245")


def panels():
    rows, columns = target_grid_shape(len(config.TARGETS))
    return plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False, constrained_layout=True)


def plot_one(root, family, architecture):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    history = pd.read_csv(root / "history_all.csv")
    figure, axes = panels()
    for axis, target in zip(axes.flat, config.TARGETS):
        values = history.loc[history.target.eq(target)]
        axis.plot(values.epoch, values.train_loss, label="Train", color=COLORS[1])
        axis.plot(values.epoch, values.val_loss, label="Validation", color=COLORS[2])
        axis.set(title=TASK_LABELS[target], xlabel="Epoch", ylabel="View-level loss")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
        if family == "regression":
            score_axis = axis.twinx()
            score_axis.plot(values.epoch, values.val_pearson_r, color=COLORS[3], label="Validation r")
            score_axis.set_ylabel("Validation Pearson r", color=COLORS[3])
            finite = values.val_pearson_r.to_numpy(float)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                pad = max(np.ptp(finite) * .15, .04)
                score_axis.set_ylim(max(-1, finite.min() - pad), min(1, finite.max() + pad))
    figure.suptitle(f"{LABELS[architecture]} | {family}")
    figure.savefig(figures / "training_history.png", dpi=180)
    plt.close(figure)
    if family == "classification":
        plot_confusion_matrices(root, config.TARGETS, run_root="runs")
        return
    figure, axes = panels()
    for axis, target in zip(axes.flat, config.TARGETS):
        frame = pd.read_csv(root / f"runs/{target}/video_predictions.csv").query("split == 'test'")
        actual, predicted = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        axis.scatter(actual, predicted, s=14, alpha=.45, color=COLORS[1], edgecolors="none")
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axis.plot([low, high], [low, high], "--", color=COLORS[0], label="Identity")
        if np.ptp(actual):
            slope, intercept = np.polyfit(actual, predicted, 1)
            x = np.array([actual.min(), actual.max()])
            axis.plot(x, slope * x + intercept, color=COLORS[2], label="Linear fit")
        axis.set(title=f"{TASK_LABELS[target]} | n={len(frame)}", xlabel=f"Measured ({TASK_UNITS[target]})",
                 ylabel=f"Predicted ({TASK_UNITS[target]})")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    for extension in ("png", "pdf"):
        figure.savefig(figures / f"test_predicted_vs_true.{extension}", dpi=180)
    plt.close(figure)


def compare():
    figures = config.OUTPUT_DIR / "figures"
    figures.mkdir(exist_ok=True)
    for family in ("classification", "regression"):
        roots = [config.BASELINES[family]] + [config.OUTPUT_DIR / family / arch for arch in config.ARCHITECTURES]
        names = ("efficientnet_b0", *config.ARCHITECTURES)
        values = []
        for architecture, root in zip(names, roots):
            frame = pd.read_csv(root / "metrics_all.csv").query("split == 'test'").copy()
            frame["architecture"] = architecture
            values.append(frame)
        for target in config.TARGETS:
            baseline = pd.read_csv(roots[0] / f"runs/efficientnet_b0/{target}/video_predictions.csv",
                                   dtype={"hospital_id": str, "video_id": str}).query("split == 'test'")
            for root in roots[1:]:
                candidate = pd.read_csv(root / f"runs/{target}/video_predictions.csv",
                                        dtype={"hospital_id": str, "video_id": str}).query("split == 'test'")
                identity = ["hospital_id", "video_id", "y_true"]
                pd.testing.assert_frame_equal(baseline.sort_values("video_id")[identity].reset_index(drop=True),
                                              candidate.sort_values("video_id")[identity].reset_index(drop=True), check_dtype=False)
        table = pd.concat(values, ignore_index=True)
        table.to_csv(config.OUTPUT_DIR / f"{family}_architecture_comparison.csv", index=False)
        metrics = (("roc_auc", "AUROC"), ("balanced_accuracy", "Balanced accuracy"), ("f1", "F1"), ("average_precision", "Average precision")) if family == "classification" else (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "R2"))
        for key, label in metrics:
            figure, axes = panels()
            for axis, target in zip(axes.flat, config.TARGETS):
                subset = table.loc[table.target.eq(target)].set_index("architecture").loc[list(names)]
                bars = axis.bar(range(4), subset[key], color=COLORS)
                axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)
                axis.set_xticks(range(4), ["EN-B0", "Histogram", "Statistics", "Small CNN"], rotation=25, ha="right")
                axis.set(title=TASK_LABELS[target], ylabel=f"{label} ({TASK_UNITS[target]})" if key in ("mae", "rmse") else label)
                if family == "classification":
                    axis.set_ylim(0, 1.12)
                axis.axhline(0, color=COLORS[0], linewidth=.7)
                axis.grid(axis="y", alpha=.2)
                axis.set_axisbelow(True)
            figure.suptitle(f"{family.capitalize()} | same 12h held-out videos | {label}")
            for extension in ("png", "pdf"):
                figure.savefig(figures / f"{family}_{key}_comparison.{extension}", dpi=180)
            plt.close(figure)
