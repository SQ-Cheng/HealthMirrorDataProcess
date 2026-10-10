"""Five-model regression comparisons, with exact held-out cohort audits."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.exp2_face_architecture_ablation.plots import panels, plot_one, LABELS as HEAD_LABELS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from study.exp6_face_pair_dinov3_regression.plots import plot_head
from study.exp6_face_pair_lab_delta.plot_results import DISPLAY
from . import run_frozen_en_regression_controls as control


LABELS = ("Fine-tuned EN-B0 (head32)", "Frozen EN-B0 (head32)", "Frozen EN-B0 (head64)",
          "Frozen DINOv3-S (head32)", "Frozen DINOv3-S (head64)")
TICKS = ("EN-B0\nfine-tuned", "EN-B0\nfrozen32", "EN-B0\nfrozen64", "DINOv3\nhead32", "DINOv3\nhead64")
COLORS = ("#64737C", "#D18C34", "#B44F59", "#278245", "#2878B5")


def model_roots(family):
    if family == "exp2":
        directory = control.exp2.config.HERE
        roots = [control.exp2.BASELINE, *(control.result_root(family, width) for width in (32, 64))]
        roots.extend(directory / ("outputs/main24h_frame_loss" + ("_head64" if width == 64 else "")) /
                     "regression/dinov3_vits16_frozen" for width in (32, 64))
    else:
        roots = [control.exp6.config.BASELINE, *(control.result_root(family, width) for width in (32, 64)),
                 control.exp6.config.OUTPUT / "head32", control.exp6.config.OUTPUT / "head64"]
    return roots


def compare(family, targets, roots=None, output=None):
    roots = model_roots(family) if roots is None else roots
    output = control.ROOTS[family].parent if output is None else output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tables = [pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(targets)].copy() for root in roots]
    filename = "video_predictions.csv" if family == "exp2" else "pair_predictions.csv"
    identifiers = (["hospital_id", "video_id", "y_true"] if family == "exp2" else
                   ["hospital_id", "pair_id", "first_video_id", "second_video_id", "y_true"])
    audits = []
    for target in targets:
        prediction = []
        for i, root in enumerate(roots):
            run = root / (f"runs/efficientnet_b0/{target}" if family == "exp2" and i == 0 else f"runs/{target}")
            options = {"dtype": {"hospital_id": str, "video_id": str}}
            if family == "exp6":
                options["float_precision"] = "round_trip"
            frame = pd.read_csv(run / filename, **options).query("split == 'test'")
            frame = frame.sort_values("video_id" if family == "exp2" else "pair_id").reset_index(drop=True)
            if not frame.frame_count.eq(20).all():
                raise RuntimeError("Five-model comparison has inconsistent frame counts")
            prediction.append(frame)
            tables[i].loc[target, "explained_variance"] = explained_variance_score(frame.y_true, frame.y_pred)
        for frame in prediction[1:]:
            pd.testing.assert_frame_equal(prediction[0][identifiers], frame[identifiers], check_dtype=False)
        audits.append({"target": target, "test_samples": len(prediction[0]), "test_patients": prediction[0].hospital_id.nunique(),
                       "identical_five_model_test_cohorts": True})
    pd.DataFrame(audits).to_csv(output / "five_model_test_audit.csv", index=False)
    pd.concat([table.assign(model=label).reset_index() for table, label in zip(tables, LABELS)], ignore_index=True).to_csv(
        output / "five_model_test_comparison.csv", index=False)
    fields = [("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"),
              ("r2", "$R^2$"), ("explained_variance", "Explained variance")]
    if family == "exp6":
        fields.extend([("direction_balanced_accuracy", "Direction balanced accuracy"), ("direction_roc_auc", "Direction AUROC")])
    labels = TASK_LABELS if family == "exp2" else DISPLAY
    units = TASK_UNITS if family == "exp2" else control.exp6.config.baseline.TARGET_UNITS
    for metric, label in fields:
        figure, axes = panels(targets)
        for axis in axes.flat[len(targets):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, targets):
            values = [table.loc[target, metric] for table in tables]
            bars = axis.bar(range(5), values, color=COLORS)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
            axis.set_xticks(range(5), TICKS, fontsize=8, rotation=15)
            unit = f" ({units[target]})" if metric in ("mae", "rmse") else ""
            axis.set(title=labels[target], ylabel=label + unit)
            axis.margins(y=.2)
            axis.axhline(.5 if metric.startswith("direction_") else 0, color=COLORS[0], linestyle="--", linewidth=.7)
            if metric.startswith("direction_"):
                axis.set_ylim(0, 1.12)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle(f"{'Single-face lab values' if family == 'exp2' else 'Paired-face lab changes'} | same 24h test cohort | {label}")
        for extension in ("png", "pdf"):
            figure.savefig(figures / f"five_model_{metric}_comparison.{extension}", dpi=180)
        plt.close(figure)
    print(f"[five-model-plots] {family}: {figures}", flush=True)


def plot_all(manifests):
    for family, manifest in manifests.items():
        targets = tuple(manifest["targets"])
        for width in (32, 64):
            root = control.result_root(family, width)
            if family == "exp2":
                HEAD_LABELS[control.ARCHITECTURE] = f"Frozen EfficientNet-B0 + head{width}"
                plot_one(root, "regression", control.ARCHITECTURE, targets=targets, loss_label="Frame-level SmoothL1")
            else:
                plot_head(root, width, targets, model_label="Frozen EfficientNet-B0")
        compare(family, targets)
