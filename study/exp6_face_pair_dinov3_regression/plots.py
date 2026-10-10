"""Paired test comparisons for fine-tuned EN-B0 and frozen DINO head32/head64."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape, target_grid_figsize
from study.exp6_face_pair_lab_delta.plot_results import DISPLAY
from . import config


COLORS = ("#64737C", "#278245", "#2878B5")


def panels(targets):
    rows, columns = target_grid_shape(len(targets))
    figure, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False, constrained_layout=True)
    for axis in axes.flat[len(targets):]:
        axis.set_visible(False)
    return figure, axes


def save(figure, path):
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix("." + extension), dpi=180)
    plt.close(figure)


def plot_head(root, hidden, targets, model_label="Frozen DINOv3-S"):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    history = pd.read_csv(root / "history_all.csv")
    for value, ylabel, filename in (("mae", "MAE (raw delta unit)", "training_histories"),
                                    ("loss", "Weighted frame-pair SmoothL1", "loss_history")):
        figure, axes = panels(targets)
        for axis, target in zip(axes.flat, targets):
            selected = history.loc[history.target.eq(target)]
            axis.plot(selected.global_epoch, selected[f"train_{value}"], label="Train", color=COLORS[2])
            axis.plot(selected.global_epoch, selected[f"val_{value}"], label="Validation", color="#CB6547")
            r_axis = axis.twinx()
            r_axis.plot(selected.global_epoch, selected.train_pearson_r, label="Train r", color=COLORS[1], linestyle=":")
            r_axis.plot(selected.global_epoch, selected.val_pearson_r, label="Validation r", color="#8C4B99", linestyle="-.")
            finite = selected[["train_pearson_r", "val_pearson_r"]].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                pad = max(np.ptp(finite) * .12, .04)
                r_axis.set_ylim(max(-1, finite.min() - pad), min(1, finite.max() + pad))
            axis.set(title=DISPLAY[target], xlabel="Epoch", ylabel=ylabel)
            r_axis.set_ylabel("Pearson r")
            handles, labels = axis.get_legend_handles_labels()
            r_handles, r_labels = r_axis.get_legend_handles_labels()
            axis.legend(handles + r_handles, labels + r_labels, fontsize=7)
            axis.grid(alpha=.2)
        figure.suptitle(f"{model_label} head{hidden} | laboratory delta regression")
        save(figure, figures / filename)
    figure, axes = panels(targets)
    for axis, target in zip(axes.flat, targets):
        data = pd.read_csv(root / f"runs/{target}/pair_predictions.csv").query("split == 'test'")
        actual, predicted = data.y_true.to_numpy(float), data.y_pred.to_numpy(float)
        axis.scatter(actual, predicted, s=15, alpha=.5, color=COLORS[2], edgecolors="none")
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axis.plot([low, high], [low, high], "--", color=COLORS[0], label="Identity")
        if np.ptp(actual):
            x = np.array([actual.min(), actual.max()])
            slope, intercept = np.polyfit(actual, predicted, 1)
            axis.plot(x, slope * x + intercept, color="#CB6547", label="Linear fit")
        axis.axhline(0, color=COLORS[0], linewidth=.6)
        axis.axvline(0, color=COLORS[0], linewidth=.6)
        unit = config.baseline.TARGET_UNITS[target]
        axis.set(title=f"{DISPLAY[target]} | n={len(data)}", xlabel=f"Observed delta ({unit})", ylabel=f"Predicted delta ({unit})")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.suptitle(f"{model_label} head{hidden} | held-out laboratory delta prediction")
    save(figure, figures / "observed_vs_predicted")


def plot_results(output=config.OUTPUT, baseline=config.BASELINE):
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    targets = tuple(manifest["targets"])
    labels = ("Fine-tuned EN-B0", "DINOv3 head32", "DINOv3 head64")
    roots = (baseline, output / "head32", output / "head64")
    tables = [pd.read_csv(root / "metrics_all.csv").query("split == 'test'").set_index("target").loc[list(targets)] for root in roots]
    identities = ["pair_id", "hospital_id", "first_video_id", "second_video_id", "y_true"]
    audits = []
    for target in targets:
        frames = [pd.read_csv(root / f"runs/{target}/pair_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
                  .query("split == 'test'").sort_values("pair_id").reset_index(drop=True) for root in roots]
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frames[0][identities], frame[identities], check_dtype=False)
        if any(not frame.frame_count.eq(20).all() for frame in frames):
            raise RuntimeError("Comparison frame coverage differs")
        audits.append({"target": target, "test_pairs": len(frames[0]), "test_patients": frames[0].hospital_id.nunique(),
                       "identical_pairs_labels_and_patients": True})
    pd.DataFrame(audits).to_csv(output / "paired_test_audit.csv", index=False)
    pd.concat([table.assign(model=label).reset_index() for label, table in zip(labels, tables)], ignore_index=True).to_csv(
        output / "test_comparison.csv", index=False)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"),
                          ("r2", "R2"), ("explained_variance", "Explained variance"),
                          ("direction_balanced_accuracy", "Direction balanced accuracy"),
                          ("direction_roc_auc", "Direction AUROC")):
        figure, axes = panels(targets)
        for axis, target in zip(axes.flat, targets):
            bars = axis.bar(range(3), [table.loc[target, metric] for table in tables], color=COLORS)
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)
            axis.set_xticks(range(3), labels, rotation=20)
            unit = f" ({config.baseline.TARGET_UNITS[target]})" if metric in ("mae", "rmse") else ""
            axis.set(title=DISPLAY[target], ylabel=label + unit)
            axis.margins(y=.18)
            axis.axhline(.5 if metric.startswith("direction_") else 0, color=COLORS[0], linestyle="--", linewidth=.7)
            if metric.startswith("direction_"):
                axis.set_ylim(0, 1.12)
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        figure.suptitle(f"Same Exp6 24h test pairs | {label}")
        save(figure, figures / f"{metric}_comparison")
    for width in config.HEAD_WIDTHS:
        plot_head(output / f"head{width}", width, targets)
    print(f"[plots-complete] paired three-model comparisons: {figures}", flush=True)
