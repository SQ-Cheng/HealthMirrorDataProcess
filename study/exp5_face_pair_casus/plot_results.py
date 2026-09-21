"""Generate label, training, and held-out result figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CASUS_ANALYTES, CASUS_MAX_SCORE


DISPLAY = {
    "creatinine": "Serum creatinine",
    "bilirubin": "Serum bilirubin",
    "lactate": "Lactic acid",
    "platelets": "Platelets",
}


def plot_results(output_dir):
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(output_dir / "records.csv", dtype={"hospital_id": str})
    metrics = pd.read_csv(output_dir / "metrics.csv")
    predictions = pd.read_csv(output_dir / "video_predictions.csv", dtype={"hospital_id": str})

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    split_colors = {"train": "#4C78A8", "val": "#F2A541", "test": "#59A14F"}
    bins = np.arange(-0.5, CASUS_MAX_SCORE + 1.5, 1)
    for split in ("train", "val", "test"):
        axes[0, 0].hist(
            records.loc[records.split.eq(split), "casus_score"], bins=bins,
            alpha=0.5, label=split, color=split_colors[split],
        )
    axes[0, 0].set(title="Partial-CASUS distribution", xlabel="Score (0-16)", ylabel="Videos")
    axes[0, 0].legend()

    for axis, analyte in zip(axes.flat[1:5], CASUS_ANALYTES):
        table = pd.crosstab(records[f"{analyte}_casus_points"], records.split).reindex(range(5), fill_value=0)
        bottom = np.zeros(5)
        for split in ("train", "val", "test"):
            values = table[split].to_numpy() if split in table else np.zeros(5)
            axis.bar(range(5), values, bottom=bottom, label=split, color=split_colors[split])
            bottom += values
        axis.set(title=DISPLAY[analyte], xlabel="CASUS points", ylabel="Videos", xticks=range(5))
        axis.grid(axis="y", alpha=0.2)
    lab_span = records.lab_time_span_hours
    axes[1, 2].hist(lab_span, bins=20, color="#B07AA1", alpha=0.85)
    axes[1, 2].axvline(lab_span.median(), color="#333", linestyle="--", label=f"median={lab_span.median():.1f} h")
    axes[1, 2].set(title="Time span among four matched labs", xlabel="Hours", ylabel="Videos")
    axes[1, 2].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle("Four-laboratory partial-CASUS label audit")
    figure.tight_layout()
    figure.savefig(figure_dir / "casus_label_definition.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    test_metric = metrics[metrics.split.eq("test")].iloc[0]
    test = predictions[predictions.split.eq("test")]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bars = axes[0].bar(
        ["MAE", "RMSE"], [test_metric.mae_points, test_metric.rmse_points],
        color=("#4C78A8", "#E15759"),
    )
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].set(title="Test error", ylabel="CASUS points")
    names = ["R2", "EV", "Pearson", "Spearman"]
    values = [test_metric.r2, test_metric.explained_variance, test_metric.pearson_r, test_metric.spearman_r]
    bars = axes[1].bar(names, values, color=("#59A14F", "#F28E2B", "#76B7B2", "#B07AA1"))
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[1].axhline(0, color="#666", linestyle=":")
    axes[1].set_title("Test fit")
    axes[2].scatter(test.y_true, test.y_pred, s=28, alpha=0.75, color="#2F6B8A")
    axes[2].plot([0, CASUS_MAX_SCORE], [0, CASUS_MAX_SCORE], "--", color="#555")
    axes[2].set(
        xlim=(-0.5, CASUS_MAX_SCORE + 0.5), ylim=(-0.5, CASUS_MAX_SCORE + 0.5),
        xlabel="True partial-CASUS", ylabel="Predicted partial-CASUS",
        title="Held-out patients",
    )
    figure.suptitle("Paired pre/postoperative face partial-CASUS results")
    figure.tight_layout()
    figure.savefig(figure_dir / "results_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[plots] saved figures under {figure_dir}", flush=True)
