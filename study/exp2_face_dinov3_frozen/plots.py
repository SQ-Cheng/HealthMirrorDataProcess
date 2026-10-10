"""Head histories, predictions, and paired frozen-versus-fine-tuned comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_architecture_ablation.plots import plot_one, panels
from study.exp2_face_architecture_ablation import plots as shared
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS

from . import config


def plot_main_regression(root, baseline, targets, *, head_hidden=32, head32_root=None):
    """Compare video predictions only after proving complete test identity."""
    shared.LABELS[config.ARCHITECTURE] = f"Frozen DINOv3-S + head{head_hidden}"
    plot_one(root, "regression", config.ARCHITECTURE, targets=targets,
             loss_label="Frame-level SmoothL1")
    figures = root / "figures"
    old = pd.read_csv(baseline / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(targets)].copy()
    new = pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(targets)].copy()
    head32 = (pd.read_csv(head32_root / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(targets)].copy()
              if head32_root is not None else None)
    counts = []
    for target in targets:
        options = {"dtype": {"hospital_id": str, "video_id": str}}
        a = pd.read_csv(baseline / f"runs/efficientnet_b0/{target}/video_predictions.csv", **options).query("split == 'test'")
        b = pd.read_csv(root / f"runs/{target}/video_predictions.csv", **options).query("split == 'test'")
        a = a.sort_values("video_id").reset_index(drop=True)
        b = b.sort_values("video_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["hospital_id", "video_id", "y_true"]],
                                      b[["hospital_id", "video_id", "y_true"]], check_dtype=False)
        if not a.frame_count.eq(20).all() or not b.frame_count.eq(20).all():
            raise RuntimeError("Comparison must use twenty original frames per video")
        for frame, metrics in ((a, old), (b, new)):
            metrics.loc[target, "explained_variance"] = explained_variance_score(frame.y_true, frame.y_pred)
        if head32 is not None:
            c = pd.read_csv(head32_root / f"runs/{target}/video_predictions.csv", **options).query("split == 'test'")
            c = c.sort_values("video_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(a[["hospital_id", "video_id", "y_true"]],
                                          c[["hospital_id", "video_id", "y_true"]], check_dtype=False)
            if not c.frame_count.eq(20).all():
                raise RuntimeError("Head32 comparison has different frame coverage")
            head32.loc[target, "explained_variance"] = explained_variance_score(c.y_true, c.y_pred)
        counts.append({"target": target, "test_videos": len(a), "test_patients": a.hospital_id.nunique(),
                       "identical_test_labels_and_patients": True})
    suffix = "_dinov3" if head32 is None else "_dinov3_head64"
    comparison = old.join(new, lsuffix="_efficientnet", rsuffix=suffix)
    if head32 is not None:
        comparison = comparison.join(head32.add_suffix("_dinov3_head32"))
    comparison.to_csv(root / "paired_baseline_comparison.csv")
    pd.DataFrame(counts).to_csv(root / "paired_test_audit.csv", index=False)
    fields = (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"),
              ("r2", "R2"), ("explained_variance", "Explained variance"))
    for metric, label in fields:
        figure, axes = panels(targets)
        for axis in axes.flat[len(targets):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, targets):
            values = [old.loc[target, metric], new.loc[target, metric]]
            labels = ["Fine-tuned EN-B0", f"DINOv3 head{head_hidden}"]
            colors = ["#64737C", "#2878B5"]
            if head32 is not None:
                values.insert(1, head32.loc[target, metric])
                labels.insert(1, "DINOv3 head32")
                colors.insert(1, "#278245")
            positions = range(len(values))
            bars = axis.bar(positions, values, color=colors)
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            axis.set_xticks(positions, labels, rotation=15)
            unit = f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""
            axis.set(title=f"{TASK_LABELS[target]} | n={counts[list(targets).index(target)]['test_videos']}", ylabel=label + unit)
            axis.margins(y=.18)
            axis.axhline(0, color="#64737C", linewidth=.7)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle(f"Regression | same 24h held-out videos, 20 frames and frame-level loss | {label}")
        for extension in ("png", "pdf"):
            figure.savefig(figures / f"{metric}_baseline_comparison.{extension}", dpi=180)
        plt.close(figure)


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
