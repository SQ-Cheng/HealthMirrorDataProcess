"""Generate pair-level classification diagnostics after Exp6 training."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp6_face_pair_lab_delta.plot_results import DISPLAY


def plot_results(output_dir):
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    targets = [target for target in DISPLAY if target in set(metrics.target)]
    test = metrics.loc[metrics.split.eq("test")].set_index("target").loc[targets]
    if len(test) != len(targets):
        raise AssertionError("Missing test tasks")

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, metric, title, chance in (
        (axes[0, 0], "balanced_accuracy", "Balanced accuracy", 0.5),
        (axes[0, 1], "roc_auc", "ROC AUC", 0.5),
        (axes[1, 0], "average_precision", "Average precision", None),
        (axes[1, 1], "sensitivity", "Sensitivity", None),
    ):
        values = test[metric].to_numpy(float)
        axis.barh([DISPLAY[target] for target in targets], values, color="#287c78")
        if chance is not None:
            axis.axvline(chance, color="#8c4848", linestyle="--", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        for y, value in enumerate(values):
            axis.text(min(value + 0.015, 0.92), y, f"{value:.3f}", va="center", fontsize=8)
    figure.suptitle("Exp6 paired-face laboratory direction: test performance")
    figure.tight_layout()
    figure.savefig(figure_dir / "test_performance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    rows, columns = target_grid_shape(len(targets))
    size = target_grid_figsize(rows, columns)
    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    for axis, target in zip(axes.flat, targets):
        history = pd.read_csv(output_dir / "runs" / target / "history.csv")
        axis.plot(history.global_epoch, history.train_loss, label="Train BCE", color="#2878b5")
        axis.plot(history.global_epoch, history.val_loss, label="Val BCE", color="#d95f52")
        right = axis.twinx()
        right.plot(history.global_epoch, history.val_roc_auc, color="#268450",
                   linestyle=":", label="Val AUC")
        right.plot(history.global_epoch, history.val_balanced_accuracy,
                   color="#9458a5", linestyle="-.", label="Val bACC")
        right.set_ylim(0, 1)
        boundaries = history.loc[history.stage.ne(history.stage.shift()), "global_epoch"].iloc[1:]
        if len(boundaries):
            axis.axvline(boundaries.iloc[0] - 0.5, color="#777777", linestyle="--")
        axis.set_title(DISPLAY[target])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("BCE loss")
        right.set_ylabel("AUC / bACC")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        other_handles, other_labels = right.get_legend_handles_labels()
        axis.legend(handles + other_handles, labels + other_labels, fontsize=7)
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    figure.suptitle("Exp6 direction classification: training histories")
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(figure_dir / "training_histories.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    confusion_figure, confusion_axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    for axis, matrix_axis, target in zip(axes.flat, confusion_axes.flat, targets):
        predictions = pd.read_csv(output_dir / "runs" / target / "pair_predictions.csv")
        test_pairs = predictions.loc[predictions.split.eq("test")]
        if not test_pairs.frame_count.eq(20).all():
            raise AssertionError(f"Incomplete test-frame coverage: {target}")
        fpr, tpr, _ = roc_curve(test_pairs.label_up, test_pairs.probability_up)
        axis.plot(fpr, tpr, color="#287c78", linewidth=1.8)
        axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        axis.set_title(f"{DISPLAY[target]} (AUC {test.loc[target, 'roc_auc']:.3f})")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)

        matrix = np.array([[test.loc[target, "tn"], test.loc[target, "fp"]],
                           [test.loc[target, "fn"], test.loc[target, "tp"]]], dtype=int)
        matrix_axis.imshow(matrix, cmap="Blues", vmin=0)
        for y, x in np.ndindex(matrix.shape):
            matrix_axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
        matrix_axis.set_xticks([0, 1], ["Down", "Up"])
        matrix_axis.set_yticks([0, 1], ["Down", "Up"])
        matrix_axis.set_title(DISPLAY[target])
        matrix_axis.set_xlabel("Predicted")
        matrix_axis.set_ylabel("Observed")
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    for axis in confusion_axes.flat[len(targets):]:
        axis.axis("off")
    figure.suptitle("Exp6 test ROC curves (one vote per laboratory pair)")
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(figure_dir / "test_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    confusion_figure.suptitle("Exp6 test confusion matrices")
    confusion_figure.tight_layout(rect=(0, 0, 1, 0.975))
    confusion_figure.savefig(figure_dir / "test_confusion_matrices.png",
                             dpi=180, bbox_inches="tight")
    plt.close(confusion_figure)
    print(f"[plots-complete] directory={figure_dir}", flush=True)


if __name__ == "__main__":
    plot_results(Path(__file__).resolve().parent / "outputs")
