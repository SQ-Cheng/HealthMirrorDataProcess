"""Generate standalone and baseline-comparison figures for the SimCLR ablation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.exp2_face_pretrained_head32_regression.plot_results import (
    TASK_LABELS,
    main as plot_baseline_format,
)

from .config import OUTPUT_DIR, REFERENCE_OUTPUT_DIR, TARGETS


def plot_results(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    plot_baseline_format(output_dir)
    figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    new = pd.read_csv(output_dir / "metrics_all.csv")
    baseline = pd.read_csv(Path(REFERENCE_OUTPUT_DIR) / "metrics_all.csv")
    new = new[new.split.eq("test")].set_index("target").loc[list(TARGETS)]
    baseline = baseline[
        baseline.split.eq("test") & baseline.architecture.eq("efficientnet_b0")
    ].set_index("target").loc[list(TARGETS)]
    rows = []
    for target in TARGETS:
        row = {"target": target}
        for metric in ("mae", "rmse", "r2", "pearson_r", "spearman_r"):
            row[f"baseline_{metric}"] = baseline.loc[target, metric]
            row[f"simclr_{metric}"] = new.loc[target, metric]
            row[f"delta_{metric}"] = new.loc[target, metric] - baseline.loc[target, metric]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "simclr_vs_baseline.csv", index=False)

    x = np.arange(len(TARGETS)); width = .38
    figure, axes = plt.subplots(2, 2, figsize=(18, 10))
    for axis, metric, title in zip(axes.flat, (
        "mae", "rmse", "r2", "pearson_r",
    ), (
        "Test MAE (lower is better)", "Test RMSE (lower is better)",
        "Test R2", "Test Pearson r",
    )):
        first = baseline[metric].to_numpy(float)
        second = new[metric].to_numpy(float)
        axis.bar(x - width / 2, first, width, label="Original two-stage", color="#8C8C8C")
        axis.bar(x + width / 2, second, width, label="SimCLR three-stage", color="#2F6B8A")
        axis.axhline(0, color="#555", lw=.8)
        axis.set_title(title); axis.grid(axis="y", alpha=.25); axis.legend(fontsize=8)
        axis.set_xticks(x, [TASK_LABELS[t] for t in TARGETS], rotation=32, ha="right")
    figure.suptitle("Head32 face regression: SimCLR initialization ablation")
    figure.tight_layout(); figure.savefig(
        figure_dir / "simclr_vs_original_regression.png", dpi=190, bbox_inches="tight"
    ); plt.close(figure)

    histories = pd.read_csv(output_dir / "simclr_history_all.csv")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for target, group in histories.groupby("target", sort=False):
        axes[0].plot(group.epoch, group.loss, label=TASK_LABELS[target])
        axes[1].plot(group.epoch, group.positive_cosine, label=TASK_LABELS[target])
    axes[0].set(title="SimCLR NT-Xent", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Positive-pair cosine", xlabel="Epoch", ylabel="Cosine")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Train-split-only SimCLR histories")
    figure.tight_layout(); figure.savefig(
        figure_dir / "simclr_pretraining_history.png", dpi=180, bbox_inches="tight"
    ); plt.close(figure)
    print(f"[plots-complete] figures={figure_dir}", flush=True)


if __name__ == "__main__":
    plot_results()
