"""Generate trajectory, score-distribution, training, and test-result figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ANALYTES, TARGET_COLUMN


DISPLAY = {
    "lactate": "Lactate", "troponin": "Troponin I",
    "creatinine": "Creatinine", "total_bilirubin": "Total bilirubin",
    "platelet_count": "Platelet count", "hemoglobin": "Hemoglobin",
    "crp": "C-reactive protein", "albumin": "Albumin", "po2": "PaO2",
}


def plot_results(output_dir):
    output_dir = Path(output_dir); figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(output_dir / "records.csv", dtype={"hospital_id": str})
    trajectories = pd.read_csv(output_dir / "recovery_trajectories.csv")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    predictions = pd.read_csv(output_dir / "video_predictions.csv", dtype={"hospital_id": str})

    figure, axes = plt.subplots(2, 5, figsize=(23, 9))
    for axis, analyte in zip(axes.flat, ANALYTES):
        values = trajectories[trajectories.analyte.eq(analyte)].sort_values("progress_center")
        axis.plot(values.progress_center, values["median"], marker="o", color="#2F6B8A")
        axis.fill_between(values.progress_center, values.q25, values.q75, color="#8CB7C9", alpha=.35)
        for row in values.itertuples(index=False):
            axis.annotate(f"n={row.patients}", (row.progress_center, row.median),
                          xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6)
        axis.set(title=DISPLAY[analyte], xlabel="Postoperative progress", ylabel="Training-patient value")
        axis.grid(alpha=.2)
    axis = axes.flat[-1]
    for split, color in zip(("train", "val", "test"), ("#4C78A8", "#F2A541", "#59A14F")):
        axis.hist(records.loc[records.split.eq(split), TARGET_COLUMN], bins=16,
                  alpha=.5, label=split, color=color)
    axis.set(title="Equal-weight trajectory deviation", xlabel="Robust absolute deviation", ylabel="Postoperative videos")
    axis.legend(); axis.grid(alpha=.2)
    figure.suptitle("Training-only postoperative trajectories and derived deviation scores")
    figure.tight_layout(); figure.savefig(figure_dir / "trajectory_deviation_definition.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    test_metric = metrics[metrics.split.eq("test")].iloc[0]
    test = predictions[predictions.split.eq("test")]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bars = axes[0].bar(["MAE", "RMSE"], [test_metric.mae, test_metric.rmse], color=("#4C78A8", "#E15759"))
    axes[0].bar_label(bars, fmt="%.3f"); axes[0].set_title("Test error")
    names = ["R2", "EV", "Pearson", "Spearman"]
    values = [test_metric.r2, test_metric.explained_variance, test_metric.pearson_r, test_metric.spearman_r]
    bars = axes[1].bar(names, values, color=("#59A14F", "#F28E2B", "#76B7B2", "#B07AA1"))
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8); axes[1].axhline(0, color="#666", linestyle=":")
    axes[1].set_title("Test fit")
    axes[2].scatter(test.y_true, test.y_pred, s=28, alpha=.75, color="#2F6B8A")
    upper = max(float(test.y_true.max()), float(test.y_pred.max())) * 1.05
    axes[2].plot([0, upper], [0, upper], "--", color="#555")
    axes[2].set(xlim=(0, upper), ylim=(0, upper), xlabel="True deviation", ylabel="Predicted deviation", title="Held-out patients")
    figure.suptitle("Postoperative trajectory-deviation prediction results")
    figure.tight_layout(); figure.savefig(figure_dir / "results_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[plots] saved figures under {figure_dir}", flush=True)
