"""Exp2-style video-level figures and a paired Exp2 comparison for Exp3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp2_face_pretrained_head32_regression.plot_results import (
    SPLIT_COLORS, TASK_LABELS, TASK_UNITS,
)

from .config import SOURCE_DIR, TARGETS


VIDEO_COLOR = "#2878B5"
FACE_COLOR = "#D95F02"
EXPERIMENT_LABEL = "Face-video R3D-18 raw-value regression"


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _grid():
    rows, columns = target_grid_shape(len(TARGETS))
    return rows, columns, target_grid_figsize(rows, columns)


def _validate(output_dir, metrics):
    runs = pd.read_csv(output_dir / "run_index.csv")
    if (set(runs.target) != set(TARGETS) or len(runs) != len(TARGETS)
            or not runs.status.eq("ok").all()):
        raise RuntimeError("Exp3 tasks are incomplete or failed")
    expected = {(target, split) for target in TARGETS
                for split in ("train", "val", "test")}
    actual = set(metrics[["target", "split"]].itertuples(index=False, name=None))
    if actual != expected or len(metrics) != len(expected):
        raise RuntimeError("Exp3 metrics are incomplete or duplicated")


def _plot_training(output_dir, figures):
    rows, columns, _ = _grid()
    figure, axes = plt.subplots(rows, columns,
                                figsize=(5.8 * columns, 4.4 * rows), squeeze=False)
    for axis, target in zip(axes.flat, TARGETS):
        history = pd.read_csv(output_dir / "runs" / target / "history.csv")
        history = history.sort_values("global_epoch")
        epochs = history.global_epoch.to_numpy()
        axis.plot(epochs, history.train_loss, color=VIDEO_COLOR,
                  label="Train sampled-clip loss")
        axis.plot(epochs, history.val_loss, color="#E15759",
                  label="Validation video loss")
        axis.set_ylabel("SmoothL1 loss")
        mae_axis = axis.twinx()
        mae_axis.plot(epochs, history.val_mae, color="#F2A541",
                      label="Validation MAE")
        mae_axis.set_ylabel(f"MAE ({TASK_UNITS[target]})", color="#A96810")
        r_axis = axis.twinx()
        r_axis.spines["right"].set_position(("axes", 1.16))
        r_axis.plot(epochs, history.val_pearson_r, color="#278245",
                    label="Validation r")
        r_axis.set_ylabel("Pearson r", color="#278245")
        r_values = history.val_pearson_r.to_numpy(float)
        r_values = r_values[np.isfinite(r_values)]
        if len(r_values):
            low, high = float(r_values.min()), float(r_values.max())
            pad = max((high - low) * 0.12, 0.04)
            r_axis.set_ylim(max(-1.0, low - pad), min(1.0, high + pad))
        stage_change = history.loc[history.stage.eq("finetune"), "global_epoch"]
        if not stage_change.empty:
            axis.axvline(stage_change.min() - 0.5, color="#666666",
                         linestyle=":", linewidth=1)
        axis.set_xlabel("Epoch")
        axis.set_title(f"{TASK_LABELS[target]} | R3D-18")
        axis.grid(axis="y", alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        for overlay in (mae_axis, r_axis):
            more_handles, more_labels = overlay.get_legend_handles_labels()
            handles.extend(more_handles)
            labels.extend(more_labels)
        axis.legend(handles, labels, loc="best", fontsize=7)
    figure.suptitle(f"{EXPERIMENT_LABEL}: training history", fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.90,
                           wspace=0.85, hspace=0.50)
    figure.savefig(figures / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_metrics(output_dir, figures, metrics):
    test = metrics.loc[metrics.split.eq("test")].set_index("target").loc[list(TARGETS)]
    x = np.arange(len(TARGETS), dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, columns, labels, ylabel, limits in (
        (axes[0, 0], ("mae", "rmse"), ("MAE", "RMSE"),
         "Raw-unit prediction error", None),
        (axes[0, 1], ("pearson_r", "spearman_r"),
         ("Pearson r", "Spearman r"), "Correlation", (-1.05, 1.05)),
        (axes[1, 0], ("r2",), ("R2",), "Coefficient of determination", None),
    ):
        width = 0.8 / len(columns)
        for number, (column, label) in enumerate(zip(columns, labels)):
            position = x + (number - (len(columns) - 1) / 2) * width
            axis.bar(position, test[column], width=width,
                     color=VIDEO_COLOR, alpha=0.92 if number == 0 else 0.52,
                     hatch=None if number == 0 else "//", label=label)
        axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [TASK_LABELS[target] for target in TARGETS],
                        rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.24)
        axis.legend(fontsize=7)
    bias = []
    for target in TARGETS:
        predictions = pd.read_csv(output_dir / "runs" / target / "video_predictions.csv")
        selected = predictions.loc[predictions.split.eq("test")]
        bias.append(float((selected.y_pred - selected.y_true).mean()))
    axis = axes[1, 1]
    axis.bar(x, bias, color=VIDEO_COLOR, alpha=0.88, label="Mean signed error")
    axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
    axis.set_ylabel("Prediction bias (raw unit)")
    axis.set_xticks(x, [TASK_LABELS[target] for target in TARGETS],
                    rotation=22, ha="right")
    axis.grid(axis="y", alpha=0.24)
    axis.legend(fontsize=7)
    figure.suptitle(f"{EXPERIMENT_LABEL}: video-level test performance", fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_split_generalization(figures, metrics):
    x = np.arange(len(TARGETS), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    for axis, (metric, ylabel) in zip(
        axes.flat, (("mae", "Video-level MAE"),
                    ("pearson_r", "Video-level Pearson r"))
    ):
        for offset, split in enumerate(("train", "val", "test")):
            values = metrics.loc[metrics.split.eq(split)].set_index("target").loc[
                list(TARGETS), metric
            ]
            axis.bar(x + (offset - 1) * 0.24, values, width=0.24,
                     color=SPLIT_COLORS[split], alpha=0.88,
                     label={"train": "Train", "val": "Validation", "test": "Test"}[split])
        axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
        axis.set_xticks(x, [TASK_LABELS[target] for target in TARGETS],
                        rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title("R3D-18")
        axis.grid(axis="y", alpha=0.24)
        axis.legend()
    figure.suptitle(f"{EXPERIMENT_LABEL}: split generalization", fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "split_generalization.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_predictions_and_comparison(output_dir, figures, metrics, reference_dir):
    rows, columns, size = _grid()
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    test = metrics.loc[metrics.split.eq("test")].set_index("target")
    comparison_rows = []
    for axis, target in zip(axes.flat, TARGETS):
        predictions = pd.read_csv(
            output_dir / "runs" / target / "video_predictions.csv",
            dtype={"video_id": str, "hospital_id": str},
        )
        selected = predictions.loc[predictions.split.eq("test")]
        reference = pd.read_csv(
            reference_dir / "runs" / "efficientnet_b0" / target / "video_predictions.csv",
            dtype={"video_id": str, "hospital_id": str},
        )
        reference = reference.loc[reference.split.eq("test")]
        paired = selected.merge(
            reference[["video_id", "hospital_id", "binary_label", "y_true", "y_pred"]],
            on=["video_id", "hospital_id"], how="left", validate="one_to_one",
            suffixes=("_video", "_face"),
        )
        if (paired.y_pred_face.isna().any() or paired.binary_label.isna().any()
                or not np.allclose(paired.y_true_video, paired.y_true_face,
                                   atol=1e-7, rtol=0)):
            raise AssertionError(f"Exp2 comparison video or label mismatch: {target}")
        y_true = paired.y_true_video.to_numpy(float)
        y_pred = paired.y_pred_video.to_numpy(float)
        normal = paired.binary_label.to_numpy(int) == 0
        axis.scatter(y_true[normal], y_pred[normal], s=22, alpha=0.67,
                     color="#4C78A8", edgecolors="none", label="Normal side")
        axis.scatter(y_true[~normal], y_pred[~normal], s=22, alpha=0.72,
                     color="#E15759", edgecolors="none",
                     label="Abnormal/boundary side")
        lower = float(min(y_true.min(), y_pred.min()))
        upper = float(max(y_true.max(), y_pred.max()))
        padding = max((upper - lower) * 0.06, 0.05)
        limits = (lower - padding, upper + padding)
        axis.plot(limits, limits, color="#333333", linestyle="--", linewidth=1)
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(f"True value ({TASK_UNITS[target]})")
        axis.set_ylabel(f"Predicted value ({TASK_UNITS[target]})")
        values = test.loc[target]
        axis.set_title(
            f"{TASK_LABELS[target]} | R3D-18\n"
            f"n={int(values['n'])}, MAE={values['mae']:.3f} "
            f"{TASK_UNITS[target]}, r={values['pearson_r']:.3f}"
        )
        axis.grid(alpha=0.18)
        axis.legend(loc="best")
        comparison_rows.append({
            "target": target, "paired_test_videos": len(paired),
            "exp3_video_mae": float(np.mean(np.abs(y_true - y_pred))),
            "exp2_face_mae": float(np.mean(np.abs(y_true - paired.y_pred_face))),
        })
    figure.suptitle(f"{EXPERIMENT_LABEL}: video-level test predictions",
                    fontsize=15, y=1.002)
    figure.tight_layout()
    figure.savefig(figures / "test_predicted_vs_true.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "exp2_face_only_comparison.csv", index=False)
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    for axis, row in zip(axes.flat, comparison.itertuples(index=False)):
        axis.bar([0, 1], [row.exp2_face_mae, row.exp3_video_mae],
                 color=[FACE_COLOR, VIDEO_COLOR], width=0.6)
        axis.set_xticks([0, 1], ["Exp2 face", "Exp3 video"])
        axis.set_title(f"{TASK_LABELS[row.target]} | n={row.paired_test_videos}")
        axis.set_ylabel(f"Test MAE ({TASK_UNITS[row.target]})")
        axis.grid(axis="y", alpha=0.24)
    figure.suptitle("Same held-out videos: frame model vs continuous-clip model",
                    fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "exp2_face_only_test_mae_comparison.png",
                   dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_results(output_dir, reference_dir=SOURCE_DIR):
    output_dir = Path(output_dir)
    reference_dir = Path(reference_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    _validate(output_dir, metrics)
    _plot_training(output_dir, figures)
    _plot_metrics(output_dir, figures, metrics)
    _plot_split_generalization(figures, metrics)
    _plot_predictions_and_comparison(output_dir, figures, metrics, reference_dir)
    for obsolete in ("training_loss.png", "validation_mae.png", "validation_r.png",
                     "test_performance.png"):
        (figures / obsolete).unlink(missing_ok=True)
    print(f"[plots-complete] directory={figures}", flush=True)


if __name__ == "__main__":
    from .config import OUTPUT_DIR
    plot_results(OUTPUT_DIR)
