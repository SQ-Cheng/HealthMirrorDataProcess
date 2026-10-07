"""Head histories, predictions, and paired frozen-versus-fine-tuned comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from study.exp2_face_architecture_ablation.plots import plot_one, panels
from study.exp2_face_architecture_ablation import plots as shared
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS

from . import config


def plot_results():
    for family in ("classification", "regression"):
        root = config.HERE / "outputs" / family / config.ARCHITECTURE
        shared.LABELS[config.ARCHITECTURE] = "Frozen DINOv3-S + head32"
        plot_one(root, family, config.ARCHITECTURE)
        baseline = config.BASELINES[family]
        old = pd.read_csv(baseline / "metrics_all.csv").query("split == 'test'").set_index("target")
        new = pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target")
        for target in config.TARGETS:
            a = pd.read_csv(baseline / f"runs/efficientnet_b0/{target}/video_predictions.csv", dtype={"hospital_id": str, "video_id": str}).query("split == 'test'")
            b = pd.read_csv(root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str, "video_id": str}).query("split == 'test'")
            columns = ["hospital_id", "video_id", "y_true"]
            pd.testing.assert_frame_equal(a.sort_values("video_id")[columns].reset_index(drop=True),
                                          b.sort_values("video_id")[columns].reset_index(drop=True), check_dtype=False)
        fields = [("roc_auc", "AUROC"), ("balanced_accuracy", "Balanced accuracy")] if family == "classification" else [("mae", "MAE"), ("pearson_r", "Pearson r"), ("r2", "R2")]
        comparison = old.join(new, lsuffix="_efficientnet", rsuffix="_dinov3")
        comparison.to_csv(root / "paired_baseline_comparison.csv")
        for metric, label in fields:
            figure, axes = panels()
            for axis, target in zip(axes.flat, config.TARGETS):
                bars = axis.bar([0, 1], [old.loc[target, metric], new.loc[target, metric]], color=["#64737C", "#2878B5"])
                axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
                axis.set_xticks([0, 1], ["Fine-tuned EN-B0", "Frozen DINOv3-S"], rotation=15)
                axis.set(title=TASK_LABELS[target], ylabel=f"{label} ({TASK_UNITS[target]})" if metric == "mae" else label)
                if family == "classification":
                    axis.set_ylim(0, 1.1)
                axis.axhline(0, color="#64737C", linewidth=.7)
                axis.grid(axis="y", alpha=.2)
            figure.suptitle(f"{family.capitalize()} | same 12h split and view-level loss")
            figure.savefig(root / f"figures/{metric}_baseline_comparison.png", dpi=180)
            plt.close(figure)
