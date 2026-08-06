"""Compare range-weighted runs against the matched unweighted history baseline."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
TARGETS = ("hemoglobin_low", "oxyhemoglobin_fraction")
LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "hemoglobin_low": "Hemoglobin",
    "oxyhemoglobin_fraction": "Oxyhemoglobin fraction",
}
METRICS = (("mae", "MAE", False), ("rmse", "RMSE", False),
           ("pearson_r", "Pearson r", True), ("r2", "R2", True))


def _test_metrics(directory):
    frame = pd.read_csv(directory / "metrics_all.csv")
    frame = frame.loc[
        frame["split"].eq("test")
        & frame["architecture"].isin(ARCHITECTURES)
        & frame["target"].isin(TARGETS)
    ]
    expected = {(a, t) for a in ARCHITECTURES for t in TARGETS}
    observed = set(frame[["architecture", "target"]].itertuples(index=False, name=None))
    if observed != expected or len(frame) != len(expected):
        raise RuntimeError(f"Incomplete comparison metrics in {directory}: {observed}")
    return frame.set_index(["architecture", "target"])


def _aligned_predictions(baseline, weighted, architecture, target):
    relative = Path("runs") / architecture / target / "video_predictions.csv"
    columns = ["hospital_id", "video_id", "split", "y_true", "y_pred", "frame_count"]
    first = pd.read_csv(baseline / relative, usecols=columns).query("split == 'test'")
    second = pd.read_csv(weighted / relative, usecols=columns).query("split == 'test'")
    joined = first.merge(
        second,
        on=["hospital_id", "video_id"],
        suffixes=("_unweighted", "_weighted"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"Test video mismatch for {architecture}/{target}")
    if not np.allclose(joined["y_true_unweighted"], joined["y_true_weighted"], atol=1e-7):
        raise RuntimeError(f"Test labels differ for {architecture}/{target}")
    if not (joined["frame_count_unweighted"].eq(20).all()
            and joined["frame_count_weighted"].eq(20).all()):
        raise RuntimeError(f"Frame count differs for {architecture}/{target}")
    return joined


def _plot_weight_ranges(weighted, figure_dir):
    with open(weighted / "range_weighting.json", encoding="utf-8") as handle:
        policies = json.load(handle)["targets"]
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(14, 5), squeeze=False)
    for axis, target in zip(axes.flat, TARGETS):
        policy = policies[target]
        edges = np.asarray(policy["bin_edges"], dtype=float)
        counts = np.asarray(policy["train_video_counts"], dtype=int)
        weights = np.asarray(policy["bin_weights"], dtype=float)
        x = np.arange(len(counts))
        bars = axis.bar(x, counts, color="#4C78A8", alpha=0.85, label="Train videos")
        axis.bar_label(bars, labels=[str(value) for value in counts], fontsize=8)
        weight_axis = axis.twinx()
        weight_axis.plot(x, weights, color="#E15759", marker="o", linewidth=1.6,
                         label="Loss weight")
        for index, value in enumerate(weights):
            weight_axis.annotate(f"{value:.2f}", (index, value), xytext=(0, 6),
                                 textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(x, [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in x],
                        rotation=22, ha="right")
        axis.set_ylabel("Training videos")
        weight_axis.set_ylabel("Normalized loss weight")
        axis.set_title(LABELS[target])
        axis.grid(axis="y", alpha=0.2)
        handles_a, labels_a = axis.get_legend_handles_labels()
        handles_b, labels_b = weight_axis.get_legend_handles_labels()
        axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper center")
    figure.suptitle("Train-only raw-value ranges and loss weights", fontsize=15)
    figure.tight_layout()
    figure.savefig(figure_dir / "training_range_weights.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(baseline_dir, weighted_dir):
    baseline, weighted = Path(baseline_dir).resolve(), Path(weighted_dir).resolve()
    figure_dir = weighted / "figures" / "comparison_unweighted"
    table_dir = weighted / "comparison_unweighted"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    base_metrics, weighted_metrics = _test_metrics(baseline), _test_metrics(weighted)
    _plot_weight_ranges(weighted, figure_dir)
    groups = [(a, t) for t in TARGETS for a in ARCHITECTURES]
    labels = [f"{LABELS[t]}\n{LABELS[a]}" for a, t in groups]
    rows = []
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    x, width = np.arange(len(groups)), 0.36
    for axis, (metric, title, higher_better) in zip(axes.flat, METRICS):
        old = np.asarray([base_metrics.loc[group, metric] for group in groups], float)
        new = np.asarray([weighted_metrics.loc[group, metric] for group in groups], float)
        first = axis.bar(x - width / 2, old, width, color="#4C78A8", label="Unweighted")
        second = axis.bar(x + width / 2, new, width, color="#E15759", label="Range weighted")
        axis.bar_label(first, fmt="%.3f", fontsize=7)
        axis.bar_label(second, fmt="%.3f", fontsize=7)
        axis.set_xticks(x, labels)
        axis.set_title(f"Test {title}")
        axis.axhline(0, color="#777777", linestyle=":", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        for group, old_value, new_value in zip(groups, old, new):
            rows.append({
                "architecture": group[0], "target": group[1], "metric": metric,
                "unweighted": old_value, "range_weighted": new_value,
                "weighted_minus_unweighted": new_value - old_value,
                "favorable_change": ((new_value - old_value) if higher_better
                                     else (old_value - new_value)),
            })
    figure.suptitle("Train-range weighting vs matched unweighted baseline", fontsize=15)
    figure.tight_layout()
    figure.savefig(figure_dir / "test_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    alignment = []
    figure, axes = plt.subplots(len(TARGETS), len(ARCHITECTURES), figsize=(14, 10), squeeze=False)
    for row_index, target in enumerate(TARGETS):
        for column_index, architecture in enumerate(ARCHITECTURES):
            joined = _aligned_predictions(baseline, weighted, architecture, target)
            axis = axes[row_index, column_index]
            truth = joined["y_true_unweighted"].to_numpy(float)
            axis.scatter(truth, joined["y_pred_unweighted"], s=20, alpha=0.55,
                         color="#4C78A8", label="Unweighted")
            axis.scatter(truth, joined["y_pred_weighted"], s=20, alpha=0.55,
                         color="#E15759", marker="x", label="Range weighted")
            limits = [min(truth.min(), joined["y_pred_unweighted"].min(), joined["y_pred_weighted"].min()),
                      max(truth.max(), joined["y_pred_unweighted"].max(), joined["y_pred_weighted"].max())]
            axis.plot(limits, limits, "--", color="#555555", linewidth=1)
            axis.set_title(f"{LABELS[target]} | {LABELS[architecture]}")
            axis.set_xlabel("True value")
            axis.set_ylabel("Predicted value")
            axis.grid(alpha=0.18)
            axis.legend()
            alignment.append({"architecture": architecture, "target": target,
                              "matched_test_videos": len(joined),
                              "identical_targets": True, "frames_per_video": 20})
    figure.suptitle("Matched video-level test predictions", fontsize=15)
    figure.tight_layout()
    figure.savefig(figure_dir / "test_prediction_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(table_dir / "test_metric_comparison.csv", index=False)
    pd.DataFrame(alignment).to_csv(table_dir / "data_alignment.csv", index=False)
    print(f"Saved 3 weighted/unweighted figures to {figure_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--weighted-dir", required=True)
    args = parser.parse_args()
    main(args.baseline_dir, args.weighted_dir)
