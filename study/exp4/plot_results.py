"""Generate final figures for the selected Exp4 seed."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(output_dir):
    output_dir = Path(output_dir)
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    run_index = pd.read_csv(output_dir / "run_index.csv")
    if len(run_index) != 1 or run_index.iloc[0]["status"] != "ok":
        raise RuntimeError("Exp4 expects exactly one completed selected-seed run")
    predictions = pd.read_csv(Path(run_index.iloc[0]["run_dir"]) / "video_predictions.csv")
    test = metrics.loc[metrics.split.eq("test")]
    if len(test) != 1:
        raise RuntimeError("Exp4 expects one test metric row for the selected seed")
    test_predictions = predictions.loc[predictions.split.eq("test")]
    split = pd.read_csv(output_dir / "records.csv")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    error_names = ["MAE", "RMSE"]
    error_values = [float(test.iloc[0].mae), float(test.iloc[0].rmse)]
    bars = axes[0, 0].bar(error_names, error_values, color=("#4C78A8", "#E15759"))
    axes[0, 0].bar_label(bars, fmt="%.3f")
    axes[0, 0].set_title("Selected-seed test error")
    fit_names = ["R2", "Explained\nvariance", "Pearson r", "Spearman r"]
    fit_values = [
        float(test.iloc[0].r2),
        float(test.iloc[0].explained_variance),
        float(test.iloc[0].pearson_r),
        float(test.iloc[0].spearman_r),
    ]
    bars = axes[0, 1].bar(
        fit_names, fit_values, color=("#59A14F", "#F28E2B", "#76B7B2", "#B07AA1")
    )
    axes[0, 1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0, 1].set_title("Selected-seed test goodness of fit")
    axes[0, 1].axhline(0, color="#666", linestyle=":")
    axes[1, 0].scatter(
        test_predictions.y_true, test_predictions.y_pred,
        alpha=0.7, s=28, color="#4C78A8",
    )
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#555")
    axes[1, 0].set(
        xlim=(0, 1), ylim=(0, 1), xlabel="True recovery",
        ylabel="Predicted recovery", title="Selected seed on held-out videos",
    )
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
    pd.DataFrame({
        "metric": numeric,
        "value": [float(test.iloc[0][metric]) for metric in numeric],
    }).to_csv(output_dir / "test_metrics_summary.csv", index=False)
    print(f"[plots] saved {figure_dir / 'results_summary.png'}", flush=True)
