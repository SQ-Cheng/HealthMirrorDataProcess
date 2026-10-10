"""Native DINO baseline versus concat and gated heads on identical test videos."""

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
    names = ("Frozen DINOv3-S", "DINOv3 + FaRL concat", "DINOv3 + FaRL gated")
    for variant, label in zip(config.VARIANTS, names[1:]):
        LABELS[variant] = label
        plot_one(config.OUTPUT / f"regression/{variant}", "regression", variant, targets=config.TARGETS, loss_label="Frame-level SmoothL1")
    tables = [pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").copy() for root in roots]
    audit = []
    for target in config.TARGETS:
        predictions = [pd.read_csv(root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str, "video_id": str}).query("split == 'test'").sort_values("video_id").reset_index(drop=True) for root in roots]
        for pred in predictions[1:]:
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
            bars = axis.bar(range(3), [table.loc[target, metric] for table in tables], color=["#64737c", "#2878b5", "#278245"])
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
            axis.set_xticks(range(3), ["DINOv3", "Concat", "Gated"])
            axis.set(title=TASK_LABELS[target], ylabel=label + (f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.margins(y=.18)
            axis.axhline(0, color="#64737c", linewidth=.6)
            axis.grid(axis="y", alpha=.2)
        figure.suptitle(f"Same native224 / 24h / 20-frame test videos | {label}")
        for extension in ("png", "pdf"):
            figure.savefig(figures / f"{metric}_comparison.{extension}", dpi=180)
        plt.close(figure)
