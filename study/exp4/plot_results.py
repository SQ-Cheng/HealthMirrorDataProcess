"""Aggregate four Exp4 seeds and generate final figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(output_dir):
    output_dir = Path(output_dir)
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    predictions = pd.concat(
        [pd.read_csv(path) for path in sorted((output_dir / "runs").glob("seed_*/video_predictions.csv"))],
        ignore_index=True,
    )
    test = metrics.loc[metrics.split.eq("test")]
    test_predictions = predictions.loc[predictions.split.eq("test")]
    ensemble = test_predictions.groupby(["hospital_id", "video_id"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean")
    )
    split = pd.read_csv(output_dir / "records.csv")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    seeds = test.seed.astype(str).tolist(); x = np.arange(len(test))
    bars = axes[0, 0].bar(x - 0.18, test.mae, 0.36, label="MAE", color="#4C78A8")
    axes[0, 0].bar_label(bars, fmt="%.3f", fontsize=8)
    bars = axes[0, 0].bar(x + 0.18, test.rmse, 0.36, label="RMSE", color="#E15759")
    axes[0, 0].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0, 0].set_xticks(x, seeds); axes[0, 0].set_xlabel("Seed"); axes[0, 0].set_title("Test error"); axes[0, 0].legend()
    bars = axes[0, 1].bar(x - 0.18, test.r2, 0.36, label="R²", color="#59A14F")
    axes[0, 1].bar_label(bars, fmt="%.3f", fontsize=8)
    bars = axes[0, 1].bar(x + 0.18, test.explained_variance, 0.36, label="Explained variance", color="#F28E2B")
    axes[0, 1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0, 1].set_xticks(x, seeds); axes[0, 1].set_xlabel("Seed"); axes[0, 1].set_title("Test goodness of fit"); axes[0, 1].axhline(0, color="#666", linestyle=":"); axes[0, 1].legend()
    axes[1, 0].scatter(ensemble.y_true, ensemble.y_pred, alpha=0.7, s=28, color="#4C78A8")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#555")
    axes[1, 0].set(xlim=(0, 1), ylim=(0, 1), xlabel="True recovery", ylabel="Predicted recovery", title="Four-seed ensemble on held-out videos")
    bins = np.linspace(0, 1, 11)
    for name, color in zip(("train", "val", "test"), ("#4C78A8", "#F28E2B", "#59A14F")):
        axes[1, 1].hist(split.loc[split.split.eq(name), "recovery_score"], bins=bins,
                        density=True, histtype="step", linewidth=2, label=name, color=color)
    axes[1, 1].set(xlabel="Recovery score", ylabel="Density", title="Patient-disjoint split distributions"); axes[1, 1].legend()
    for axis in axes.flat: axis.grid(alpha=0.2)
    figure.suptitle("Exp4: postoperative recovery from facial videos", fontsize=15)
    figure.tight_layout(); figure_dir = output_dir / "figures"; figure_dir.mkdir(exist_ok=True)
    figure.savefig(figure_dir / "results_summary.png", dpi=180, bbox_inches="tight"); plt.close(figure)

    numeric = ["mae", "rmse", "r2", "explained_variance", "pearson_r", "spearman_r"]
    summary = test[numeric].agg(["mean", "std"]).T.reset_index().rename(
        columns={"index": "metric"}
    )
    summary.to_csv(output_dir / "test_metrics_seed_summary.csv", index=False)
    print(f"[plots] saved {figure_dir / 'results_summary.png'}", flush=True)
