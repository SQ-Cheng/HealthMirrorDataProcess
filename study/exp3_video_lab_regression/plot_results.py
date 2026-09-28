"""Automatic video-level Exp3 diagnostics and paired Exp2 comparison."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS

from .config import SOURCE_DIR, TARGETS


def _grid(targets):
    rows, cols = target_grid_shape(len(targets))
    return rows, cols, target_grid_figsize(rows, cols)


def plot_results(output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.read_csv(output_dir / "metrics_all.csv")
    test = all_metrics.loc[all_metrics.split.eq("test")].set_index("target").loc[list(TARGETS)]
    rows, cols, size = _grid(TARGETS)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, metric, title, reference in (
        (axes[0, 0], "mae", "Test MAE", None),
        (axes[0, 1], "rmse", "Test RMSE", None),
        (axes[1, 0], "pearson_r", "Test Pearson r", 0),
        (axes[1, 1], "r2", "Test R2", 0),
    ):
        values = test[metric].to_numpy(float)
        axis.barh([TASK_LABELS[target] for target in TARGETS], values, color="#287b76")
        if reference is not None:
            axis.axvline(reference, color="#777777", linestyle="--")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Exp3 video-level laboratory value prediction")
    figure.tight_layout()
    figure.savefig(figures / "test_performance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    for field, title, filename in (
        ("loss", "Scaled SmoothL1 loss", "training_loss.png"),
        ("mae", "Validation MAE (raw unit)", "validation_mae.png"),
        ("pearson_r", "Validation Pearson r", "validation_r.png"),
    ):
        figure, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
        for axis, target in zip(axes.flat, TARGETS):
            history = pd.read_csv(output_dir / "runs" / target / "history.csv")
            if field == "loss":
                axis.plot(history.global_epoch, history.train_loss,
                          label="Train sampled clip", color="#2878b5")
                axis.plot(history.global_epoch, history.val_loss,
                          label="Validation video", color="#d95f52")
            else:
                axis.plot(history.global_epoch, history[f"val_{field}"],
                          label="Validation video", color="#287b76")
                if field == "pearson_r":
                    values = history.val_pearson_r.to_numpy(float)
                    values = values[np.isfinite(values)]
                    if len(values):
                        pad = max((values.max() - values.min()) * 0.12, 0.04)
                        axis.set_ylim(max(-1, values.min() - pad),
                                      min(1, values.max() + pad))
            boundary = history.loc[history.stage.ne(history.stage.shift()), "global_epoch"].iloc[1:]
            if len(boundary):
                axis.axvline(boundary.iloc[0] - 0.5, linestyle=":", color="#777777")
            axis.set_title(TASK_LABELS[target])
            axis.set_xlabel("Epoch")
            axis.set_ylabel(title)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        figure.suptitle(f"Exp3 {title}")
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        figure.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    comparison_rows = []
    for axis, target in zip(axes.flat, TARGETS):
        predictions = pd.read_csv(
            output_dir / "runs" / target / "video_predictions.csv",
            dtype={"video_id": str, "hospital_id": str},
        )
        selected = predictions.loc[predictions.split.eq("test")]
        axis.scatter(selected.y_true, selected.y_pred, s=14, alpha=0.55, color="#287b76")
        low = float(min(selected.y_true.min(), selected.y_pred.min()))
        high = float(max(selected.y_true.max(), selected.y_pred.max()))
        axis.plot([low, high], [low, high], "--", linewidth=1, color="#555555")
        axis.set_title(TASK_LABELS[target])
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        axis.grid(alpha=0.2)

        reference = pd.read_csv(
            SOURCE_DIR / "runs" / "efficientnet_b0" / target / "video_predictions.csv",
            dtype={"video_id": str, "hospital_id": str},
        )
        reference = reference.loc[reference.split.eq("test")]
        paired = selected.merge(
            reference[["video_id", "hospital_id", "y_true", "y_pred"]],
            on=["video_id", "hospital_id"], how="left", validate="one_to_one",
            suffixes=("_video", "_face"),
        )
        if (paired.y_pred_face.isna().any()
                or not np.allclose(paired.y_true_video, paired.y_true_face,
                                   atol=1e-7, rtol=0)):
            raise AssertionError(f"Exp2 comparison video or label mismatch: {target}")
        comparison_rows.append({
            "target": target, "paired_test_videos": len(paired),
            "exp3_video_mae": float(np.mean(np.abs(paired.y_true_video - paired.y_pred_video))),
            "exp2_face_mae": float(np.mean(np.abs(paired.y_true_face - paired.y_pred_face))),
        })
    figure.suptitle("Exp3 held-out video predictions (one point per video)")
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(figures / "test_predicted_vs_true.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "exp2_face_only_comparison.csv", index=False)
    figure, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    for axis, row in zip(axes.flat, comparison.itertuples(index=False)):
        values = [row.exp2_face_mae, row.exp3_video_mae]
        axis.bar([0, 1], values, color=["#73808a", "#287b76"], width=0.6)
        axis.set_xticks([0, 1], ["Exp2 face", "Exp3 video"])
        axis.set_title(TASK_LABELS[row.target])
        axis.set_ylabel("Test MAE")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Same held-out videos: frame model vs continuous-clip model")
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(figures / "exp2_face_only_test_mae_comparison.png",
                   dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[plots-complete] directory={figures}", flush=True)


if __name__ == "__main__":
    from .config import OUTPUT_DIR
    plot_results(OUTPUT_DIR)
