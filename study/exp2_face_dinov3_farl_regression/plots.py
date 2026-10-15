"""Native DINO baseline versus concat and two gated heads on identical videos."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_architecture_ablation.plots import plot_one, panels, LABELS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from . import config


def plot_results():
    roots = [config.reference.RESULTS] + [config.OUTPUT / f"regression/{variant}" for variant in config.VARIANTS]
    backbone = "DINOv3-B" if config.DINO_BACKBONE == "dinov3_vitb16" else "DINOv3-S"
    names = (f"Frozen {backbone}", f"{backbone} + FaRL concat", f"{backbone} + FaRL vector gate", f"{backbone} + FaRL scalar gate")
    for variant, label in zip(config.VARIANTS, names[1:]):
        LABELS[variant] = label
        plot_one(config.OUTPUT / f"regression/{variant}", "regression", variant, targets=config.TARGETS, loss_label="Frame-level SmoothL1")
    tables = [pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").copy() for root in roots]
    audit = []
    for target in config.TARGETS:
        predictions = [pd.read_csv(root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str, "video_id": str}).query("split == 'test'").sort_values("video_id").reset_index(drop=True) for root in roots]
        for pred in predictions:
            pd.testing.assert_frame_equal(predictions[0][["hospital_id", "video_id", "y_true"]], pred[["hospital_id", "video_id", "y_true"]], check_dtype=False)
            if not pred.frame_count.eq(20).all():
                raise RuntimeError("Fusion test frame count differs")
        for prediction, table in zip(predictions, tables):
            table.loc[target, "explained_variance"] = explained_variance_score(prediction.y_true, prediction.y_pred)
        audit.append({"target": target, "identical_test_videos_patients_and_labels": True, "test_videos": len(predictions[0])})
    pd.DataFrame(audit).to_csv(config.OUTPUT / "paired_test_audit.csv", index=False)
    pd.concat([table.assign(model=name).reset_index() for table, name in zip(tables, names)], ignore_index=True).to_csv(config.OUTPUT / "test_comparison.csv", index=False)
    figures = config.OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            bars = axis.bar(range(4), [table.loc[target, metric] for table in tables], color=["#64737c", "#2878b5", "#278245", "#cb6547"])
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
            axis.set_xticks(range(4), ["DINOv3", "Concat", "Vector gate", "Scalar gate"], rotation=20, ha="right")
            axis.set(title=TASK_LABELS[target], ylabel=label + (f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.margins(y=.18)
            axis.axhline(0, color="#64737c", linewidth=.6)
            axis.grid(axis="y", alpha=.2)
        figure.suptitle(f"Same native224 / 24h / 20-frame test videos | {label}")
        figure.savefig(figures / f"{metric}_comparison.png", dpi=180)
        plt.close(figure)


def plot_backbone_comparison():
    """Eight configurations: S/B encoders, each alone and with three FaRL heads."""
    small_fusion = config.HERE / "outputs"
    small_dino = config.reference.config.HERE / "outputs/main24h_frame_loss/regression/dinov3_vits16_frozen"
    roots = [small_dino, *(small_fusion / f"regression/{variant}" for variant in config.VARIANTS),
             config.reference.RESULTS, *(config.OUTPUT / f"regression/{variant}" for variant in config.VARIANTS)]
    names = ("S / DINO only", "S / concat", "S / vector gate", "S / scalar gate",
             "B / DINO only", "B / concat", "B / vector gate", "B / scalar gate")
    tables = [pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(config.TARGETS)].copy() for root in roots]
    audit = []
    columns = ["hospital_id", "video_id", "y_true"]
    for target in config.TARGETS:
        reference = None
        for root, table in zip(roots, tables):
            predictions = pd.read_csv(root / f"runs/{target}/video_predictions.csv",
                dtype={"hospital_id": str, "video_id": str}).query("split == 'test'").sort_values("video_id").reset_index(drop=True)
            if reference is None:
                reference = predictions
            pd.testing.assert_frame_equal(reference[columns], predictions[columns], check_dtype=False)
            if not predictions.frame_count.eq(20).all():
                raise RuntimeError("Eight-model comparison requires the same twenty original frames")
            table.loc[target, "explained_variance"] = explained_variance_score(predictions.y_true, predictions.y_pred)
        audit.append({"target": target, "test_videos": len(reference), "test_patients": reference.hospital_id.nunique(),
                      "identical_eight_model_test_cohorts_and_labels": True})
    pd.concat([table.reset_index().assign(model=name) for table, name in zip(tables, names)], ignore_index=True).to_csv(
        small_fusion / "backbone_fusion_test_comparison.csv", index=False)
    pd.DataFrame(audit).to_csv(small_fusion / "backbone_fusion_test_audit.csv", index=False)
    figures = small_fusion / "figures"
    figures.mkdir(exist_ok=True)
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            for offset, group, name, color in ((-.18, tables[:4], "DINOv3-S", "#278245"), (.18, tables[4:], "DINOv3-B", "#2878b5")):
                positions = [i+offset for i in range(4)]
                bars = axis.bar(positions, [table.loc[target, metric] for table in group], width=.34, color=color, label=name)
                axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=6)
            axis.set_xticks(range(4), ["DINO only", "Concat", "Vector gate", "Scalar gate"], rotation=20, ha="right")
            axis.set(title=TASK_LABELS[target], ylabel=label+(f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.margins(y=.2); axis.axhline(0, color=".5", linewidth=.6)
            axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
        axes.flat[0].legend(fontsize=8, frameon=False)
        figure.suptitle(f"DINOv3-S/B and FaRL fusion | identical 24h cohorts, frames, views and head training | {label}")
        figure.savefig(figures / f"backbone_fusion_{metric}_comparison.png", dpi=180)
        plt.close(figure)
    print(f"[eight-model-plots] {figures}", flush=True)
