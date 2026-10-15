"""Head histories, predictions, and paired frozen-versus-fine-tuned comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_architecture_ablation.plots import plot_one, panels
from study.exp2_face_architecture_ablation import plots as shared
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS

from . import config


def plot_main_regression(root, baseline, targets, *, head_hidden=32, head32_root=None,
                         architecture=None, backbone_label="DINOv3-S"):
    """Compare video predictions only after proving complete test identity."""
    architecture = config.ARCHITECTURE if architecture is None else architecture
    shared.LABELS[architecture] = f"Frozen {backbone_label} + head{head_hidden}"
    plot_one(root, "regression", architecture, targets=targets,
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
    suffix = "_dinov3_vitb16" if backbone_label == "DINOv3-B" else ("_dinov3" if head32 is None else "_dinov3_head64")
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
            labels = ["Fine-tuned EN-B0", f"{backbone_label} head{head_hidden}"]
            colors = ["#64737C", "#2878B5"]
            if head32 is not None:
                values.insert(1, head32.loc[target, metric])
                labels.insert(1, "DINOv3-S head32")
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
        figure.savefig(figures / f"{metric}_baseline_comparison.png", dpi=180)
        plt.close(figure)


def plot_backbone_size_comparison(base_root, small_root, baseline, targets):
    """Refresh the experiment-wide S32/B32 comparison without replacing old figures."""
    import json
    output = config.HERE / "outputs"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scalers = json.loads((baseline / "target_scalers.json").read_text())["targets"]
    table, audits = [], []
    metrics = []
    for name, root in (("DINOv3-S/16 + head32", small_root), ("DINOv3-B/16 + head32", base_root)):
        frame = pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(targets)].copy()
        for target in targets:
            prediction = pd.read_csv(root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str}).query("split == 'test'")
            frame.loc[target, "explained_variance"] = explained_variance_score(prediction.y_true, prediction.y_pred)
            frame.loc[target, "mae_train_iqr"] = frame.loc[target, "mae"]/scalers[target]["iqr"]
            frame.loc[target, "rmse_train_iqr"] = frame.loc[target, "rmse"]/scalers[target]["iqr"]
        metrics.append(frame)
        table.append(frame.reset_index().assign(model=name))
    for target in targets:
        small = pd.read_csv(small_root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str}).query("split == 'test'").sort_values("video_id").reset_index(drop=True)
        base = pd.read_csv(base_root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str}).query("split == 'test'").sort_values("video_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(small[["hospital_id", "video_id", "y_true"]], base[["hospital_id", "video_id", "y_true"]], check_dtype=False)
        if not small.frame_count.eq(20).all() or not base.frame_count.eq(20).all():
            raise RuntimeError("Backbone-size comparison must use exactly 20 original-view frames")
        audits.append({"target": target, "videos": len(base), "patients": base.hospital_id.nunique(),
                       "identical_test_patients_labels_frames": True})
    pd.concat(table, ignore_index=True).to_csv(output / "backbone_size_test_comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "backbone_size_test_audit.csv", index=False)
    figure, axes = plt.subplots(2, 3, figsize=(17, 12), constrained_layout=True)
    fields = (("pearson_r", "Pearson r"), ("spearman_r", "Spearman rho"), ("r2", "R squared"),
              ("explained_variance", "Explained variance"), ("mae_train_iqr", "MAE / train IQR"),
              ("rmse_train_iqr", "RMSE / train IQR"))
    names = ("DINOv3-S/16 + head32", "DINOv3-B/16 + head32")
    for axis, (column, title) in zip(axes.flat, fields):
        for i, (frame, name, color) in enumerate(zip(metrics, names, ("#278245", "#2878B5"))):
            axis.barh(np.arange(len(targets))+(i-.5)*.34, frame[column], height=.31, label=name, color=color)
        axis.set_yticks(np.arange(len(targets)), [TASK_LABELS[target] for target in targets])
        axis.invert_yaxis(); axis.set_title(title)
        axis.grid(axis="x", alpha=.15); axis.set_axisbelow(True)
        axis.axvline(0, color=".5", linewidth=.7)
    axes.flat[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Frozen DINOv3 backbone size | identical 24h cohort, 20 frames, 5 views and head32 schedule")
    figure.savefig(figures / "backbone_size_comparison.png", dpi=180)
    plt.close(figure)
    print(f"[backbone-size-plots] {figures / 'backbone_size_comparison.png'}", flush=True)


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
