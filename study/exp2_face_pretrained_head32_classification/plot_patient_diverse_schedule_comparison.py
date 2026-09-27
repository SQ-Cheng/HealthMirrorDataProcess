"""Compare the face-only binary 30/40 ablation with its binary baseline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp2_binary_classification_common.plot_results import DISPLAY


def _run_dir(root, target):
    return root / "runs" / "efficientnet_b0" / target


def _metrics(root, targets):
    test = pd.read_csv(root / "metrics_all.csv").query("split == 'test'")
    if len(test) != len(targets) or set(test.target) != set(targets):
        raise AssertionError(f"Incomplete test metrics: {root}")
    return test.set_index("target").loc[list(targets)]


def _check_predictions(baseline_dir, variant_dir, target):
    frames = [pd.read_csv(
        _run_dir(root, target) / "video_predictions.csv",
        dtype={"hospital_id": str, "video_id": str},
    ).sort_values("video_id").reset_index(drop=True)
              for root in (baseline_dir, variant_dir)]
    baseline, variant = frames
    if len(baseline) != len(variant):
        raise AssertionError(f"Different video counts: {target}")
    for key in ("hospital_id", "video_id", "split", "y_true"):
        if not baseline[key].astype(str).equals(variant[key].astype(str)):
            raise AssertionError(f"Different {key} between variants: {target}")
    if not baseline.input_count.eq(20).all() or not variant.input_count.eq(20).all():
        raise AssertionError(f"Test-frame coverage differs: {target}")


def _metric_bars(table, targets, metric, label, path):
    rows, cols = target_grid_shape(len(targets))
    fig, axes = plt.subplots(rows, cols, figsize=target_grid_figsize(rows, cols), squeeze=False)
    for axis, target in zip(axes.flat, targets):
        values = [table.loc[target, f"baseline_{metric}"],
                  table.loc[target, f"ablation_{metric}"]]
        axis.bar([0, 1], values, color=["#7b8490", "#278079"], width=0.62)
        axis.set_xticks([0, 1], ["Baseline", "Patient diverse\n30/40"])
        axis.set_ylim(0, 1)
        axis.set_title(DISPLAY.get(target, target))
        axis.grid(axis="y", alpha=0.2)
        for x, value in enumerate(values):
            axis.annotate(f"{value:.3f}", (x, value), xytext=(0, 4),
                          textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle(f"Exp2 face-only classification: test {label}")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _history(baseline_dir, variant_dir, targets, metric, label, path):
    rows, cols = target_grid_shape(len(targets))
    fig, axes = plt.subplots(rows, cols, figsize=target_grid_figsize(rows, cols), squeeze=False)
    for axis, target in zip(axes.flat, targets):
        for root, text, color in ((baseline_dir, "Baseline", "#7b8490"),
                                  (variant_dir, "Patient diverse 30/40", "#278079")):
            frame = pd.read_csv(_run_dir(root, target) / "history.csv")
            axis.plot(frame.global_epoch, frame[f"val_{metric}"],
                      label=text, color=color)
            boundary = frame.loc[frame.stage.ne(frame.stage.shift()), "global_epoch"].iloc[1:]
            if len(boundary):
                axis.axvline(boundary.iloc[0] - 0.5, color=color,
                             linestyle=":", alpha=0.7)
        axis.set_title(DISPLAY.get(target, target))
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle(f"Exp2 face-only classification: validation {label}")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(baseline_dir, variant_dir, targets):
    baseline_dir, variant_dir = Path(baseline_dir), Path(variant_dir)
    targets = tuple(targets)
    baseline = _metrics(baseline_dir, targets)
    variant = _metrics(variant_dir, targets)
    for target in targets:
        _check_predictions(baseline_dir, variant_dir, target)
    if not baseline.n.equals(variant.n):
        raise AssertionError("Comparison test counts differ")
    table = pd.DataFrame({"target": targets, "test_videos": baseline.n.to_numpy(int)}).set_index("target")
    for metric in ("balanced_accuracy", "roc_auc", "f1", "average_precision"):
        table[f"baseline_{metric}"] = baseline[metric]
        table[f"ablation_{metric}"] = variant[metric]
        table[f"delta_{metric}"] = variant[metric] - baseline[metric]
    table.reset_index().to_csv(variant_dir / "baseline_comparison.csv", index=False)
    figure_dir = variant_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, label, stem in (("balanced_accuracy", "bACC", "bacc"),
                                ("roc_auc", "ROC AUC", "auc")):
        _metric_bars(table, targets, metric, label,
                     figure_dir / f"test_{stem}_comparison.png")
        _history(baseline_dir, variant_dir, targets, metric, label,
                 figure_dir / f"validation_{stem}_history.png")
    print(f"[plots-complete] directory={figure_dir}", flush=True)


if __name__ == "__main__":
    from study.exp2_binary_classification_common.engine import EXPERIMENT_DIRS, TARGETS
    base = EXPERIMENT_DIRS["face_only"] / "outputs"
    plot_comparison(base, base / "ablations" / "patient_diverse_schedule_30_40", TARGETS)
