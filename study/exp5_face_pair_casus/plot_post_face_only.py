"""Plot post-face-only CASUS results and compare them with the paired protocol."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import CASUS_MAX_SCORE


def plot_post_face_only(run_dir):
    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    predictions = pd.read_csv(
        run_dir / "video_predictions.csv", dtype={"hospital_id": str},
    )
    test_metric = metrics.loc[metrics.split.eq("test")].iloc[0]
    test = predictions.loc[predictions.split.eq("test")]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bars = axes[0].bar(
        ["MAE", "RMSE"], [test_metric.mae_points, test_metric.rmse_points],
        color=("#4C78A8", "#E15759"),
    )
    axes[0].bar_label(bars, fmt="%.3f")
    axes[0].set(title="Test error", ylabel="CASUS points")
    names = ["R2", "EV", "Pearson", "Spearman"]
    values = [
        test_metric.r2, test_metric.explained_variance,
        test_metric.pearson_r, test_metric.spearman_r,
    ]
    bars = axes[1].bar(
        names, values, color=("#59A14F", "#F28E2B", "#76B7B2", "#B07AA1"),
    )
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[1].axhline(0, color="#666", linestyle=":")
    axes[1].set_title("Test fit")
    axes[2].scatter(
        test.y_true, test.y_pred, s=28, alpha=0.75, color="#2F6B8A",
    )
    axes[2].plot([0, CASUS_MAX_SCORE], [0, CASUS_MAX_SCORE], "--", color="#555")
    axes[2].set(
        xlim=(-0.5, CASUS_MAX_SCORE + 0.5),
        ylim=(-0.5, CASUS_MAX_SCORE + 0.5),
        xlabel="True partial-CASUS", ylabel="Predicted partial-CASUS",
        title=f"Held-out patients (n={test.hospital_id.nunique()})",
    )
    figure.suptitle("Post-face-only partial-CASUS results")
    figure.tight_layout()
    figure.savefig(figure_dir / "results_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    # Different cohort and split from the paired protocol, so a direct model
    # comparison figure would be statistically misleading.
    (figure_dir / "paired_vs_post_face_only.png").unlink(missing_ok=True)
    print(f"[plots] saved post-face-only figures under {figure_dir}", flush=True)
