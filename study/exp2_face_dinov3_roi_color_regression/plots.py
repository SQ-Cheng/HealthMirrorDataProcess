"""Histories, original-unit test scatter and paired performance bars."""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from study.exp1_roi_color_regression.plots import panels, save
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from .run import OUTPUT, reference


def plot_results():
    targets = reference.TARGETS
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    test = pd.read_csv(OUTPUT / "metrics_all.csv").query("split == 'test'")
    test.to_csv(OUTPUT / "test_comparison.csv", index=False)
    audit = []
    for target in targets:
        prediction = pd.read_csv(OUTPUT / f"runs/{target}/video_predictions.csv").query("split == 'test'")
        audit.append({"target": target, "test_videos": len(prediction), "test_patients": prediction.hospital_id.nunique(),
                      "matched_test_frames": int(prediction.frame_count.sum()), "identical_videos_labels_frames": True,
                      "training_cohort_identical": False})
    pd.DataFrame(audit).to_csv(OUTPUT / "paired_test_audit.csv", index=False)
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        fig, axes = panels(targets)
        for axis, target in zip(axes.flat, targets):
            part = test.loc[test.target.eq(target)].set_index("model")
            bars = axis.bar([0, 1], [part.loc[name, metric] for name in ("DINO-only (matched frames)", "DINO + ROI")], color=["#64737c", "#2878b5"])
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
            axis.set_xticks([0, 1], ["DINO-only", "DINO + ROI"])
            axis.set(title=TASK_LABELS[target], ylabel=label + (f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.margins(y=.2)
            axis.axhline(0, color="#888888", linewidth=.6)
            axis.grid(axis="y", alpha=.2)
        fig.suptitle(f"Matched original-view test videos and frames | {label}")
        save(fig, figures / f"{metric}_comparison")
    fig, axes = panels(targets)
    for axis, target in zip(axes.flat, targets):
        history = pd.read_csv(OUTPUT / f"runs/{target}/history.csv")
        axis.plot(history.epoch, history.train_loss, color="#2878b5", label="Train loss")
        axis.plot(history.epoch, history.val_loss, color="#cb6547", label="Val loss")
        twin = axis.twinx()
        twin.plot(history.epoch, history.train_pearson_r, color="#278245", linestyle="--", label="Train r")
        twin.plot(history.epoch, history.val_pearson_r, color="#278245", label="Val r")
        values = history[["train_pearson_r", "val_pearson_r"]].to_numpy().flatten()
        values = values[np.isfinite(values)]
        if len(values):
            pad = max(np.ptp(values) * .1, .04)
            twin.set_ylim(max(-1, values.min() - pad), min(1, values.max() + pad))
        axis.set(title=TASK_LABELS[target], xlabel="Epoch", ylabel="SmoothL1")
        twin.set_ylabel("Pearson r")
        lines = axis.get_lines() + twin.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=7)
        axis.grid(alpha=.2)
    fig.suptitle("Frozen DINO + ROI | five-view training history")
    save(fig, figures / "training_history")
    fig, axes = panels(targets)
    for axis, target in zip(axes.flat, targets):
        frame = pd.read_csv(OUTPUT / f"runs/{target}/video_predictions.csv").query("split == 'test'")
        actual, predicted = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
        axis.scatter(actual, predicted, s=13, alpha=.5, color="#2878b5", edgecolors="none")
        lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axis.plot([lo, hi], [lo, hi], "--", color="#64737c", label="Identity")
        if np.ptp(actual):
            slope, intercept = np.polyfit(actual, predicted, 1)
            span = np.array([actual.min(), actual.max()])
            axis.plot(span, slope * span + intercept, color="#cb6547", label="Linear fit")
        axis.set(title=f"{TASK_LABELS[target]} | n={len(frame)}", xlabel=f"Measured ({TASK_UNITS[target]})", ylabel=f"Predicted ({TASK_UNITS[target]})")
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    fig.suptitle("DINO + ROI | original-view video-level test predictions")
    save(fig, figures / "test_predicted_vs_true")
    print(f"[plots-complete] {figures}", flush=True)
