"""Generate trajectory, score-distribution, training, and test-result figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ANALYTES


DISPLAY = {
    "lactate": "Lactate", "blood_gas_hb": "Blood-gas Hb",
    "troponin": "Troponin", "o2hb_fraction": "O2Hb fraction",
    "glucose": "Glucose",
}


def plot_results(output_dir):
    output_dir = Path(output_dir); figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(output_dir / "records.csv", dtype={"hospital_id": str})
    trajectories = pd.read_csv(output_dir / "recovery_trajectories.csv")
    metrics = pd.read_csv(output_dir / "metrics.csv")
    predictions = pd.read_csv(output_dir / "video_predictions.csv", dtype={"hospital_id": str})

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
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
        axis.hist(records.loc[records.split.eq(split), "recovery_score"], bins=np.linspace(0, 1, 16),
                  alpha=.5, label=split, color=color)
    axis.set(title="Equal-weight recovery score", xlabel="Score", ylabel="Postoperative videos")
    axis.legend(); axis.grid(alpha=.2)
    figure.suptitle("Training-only average trajectories and derived recovery scores")
    figure.tight_layout(); figure.savefig(figure_dir / "recovery_score_definition.png", dpi=180, bbox_inches="tight")
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
    axes[2].plot([0, 1], [0, 1], "--", color="#555")
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="True score", ylabel="Predicted score", title="Held-out patients")
    figure.suptitle("Paired pre/postoperative face recovery results")
    figure.tight_layout(); figure.savefig(figure_dir / "results_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[plots] saved figures under {figure_dir}", flush=True)
