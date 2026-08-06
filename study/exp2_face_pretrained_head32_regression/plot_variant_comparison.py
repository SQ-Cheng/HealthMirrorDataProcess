"""Plot controlled comparisons between two 20-frame regression experiments."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
TARGETS = ("hemoglobin_low", "po2_low")
TARGET_LABELS = {
    "hemoglobin_low": "Hemoglobin",
    "po2_low": "PO2",
}
METRICS = (
    ("mae", "MAE", False),
    ("rmse", "RMSE", False),
    ("pearson_r", "Pearson r", True),
    ("r2", "R2", True),
)
COLORS = ("#2878B5", "#D95F02")


def style_plots():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_test_metrics(output_dir):
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    metrics = metrics.loc[
        metrics["split"].eq("test")
        & metrics["target"].isin(TARGETS)
        & metrics["architecture"].isin(ARCHITECTURES)
    ].copy()
    expected = {(architecture, target) for architecture in ARCHITECTURES for target in TARGETS}
    observed = set(metrics[["architecture", "target"]].itertuples(index=False, name=None))
    if observed != expected or len(metrics) != len(expected):
        raise RuntimeError(
            f"Incomplete or duplicate test metrics in {output_dir}: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    return metrics.set_index(["architecture", "target"]).sort_index()


def prediction_path(output_dir, architecture, target):
    return output_dir / "runs" / architecture / target / "video_predictions.csv"


def load_aligned_test_predictions(baseline_dir, candidate_dir, architecture, target):
    keys = ["hospital_id", "video_id"]
    columns = keys + ["y_true", "y_pred", "frame_count"]
    baseline = pd.read_csv(prediction_path(baseline_dir, architecture, target), usecols=columns)
    candidate = pd.read_csv(prediction_path(candidate_dir, architecture, target), usecols=columns)
    baseline_split = pd.read_csv(
        prediction_path(baseline_dir, architecture, target), usecols=["split"]
    )["split"]
    candidate_split = pd.read_csv(
        prediction_path(candidate_dir, architecture, target), usecols=["split"]
    )["split"]
    baseline = baseline.loc[baseline_split.eq("test")].copy()
    candidate = candidate.loc[candidate_split.eq("test")].copy()
    if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise RuntimeError(f"Duplicate test videos for {architecture}/{target}")
    aligned = baseline.merge(
        candidate,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        raise RuntimeError(f"Test video sets differ for {architecture}/{target}")
    if not np.allclose(
        aligned["y_true_baseline"], aligned["y_true_candidate"], rtol=0, atol=1e-7
    ):
        raise RuntimeError(f"Test labels differ for {architecture}/{target}")
    if not (
        aligned["frame_count_baseline"].eq(20).all()
        and aligned["frame_count_candidate"].eq(20).all()
    ):
        raise RuntimeError(f"Unexpected test frame count for {architecture}/{target}")
    return aligned.drop(columns="_merge").sort_values(keys).reset_index(drop=True)


def build_metric_table(baseline, candidate, baseline_label, candidate_label):
    rows = []
    for architecture in ARCHITECTURES:
        for target in TARGETS:
            for metric, metric_label, higher_is_better in METRICS:
                baseline_value = float(baseline.loc[(architecture, target), metric])
                candidate_value = float(candidate.loc[(architecture, target), metric])
                delta = candidate_value - baseline_value
                rows.append(
                    {
                        "architecture": architecture,
                        "target": target,
                        "metric": metric,
                        "metric_label": metric_label,
                        "higher_is_better": higher_is_better,
                        "baseline_label": baseline_label,
                        "candidate_label": candidate_label,
                        "baseline_value": baseline_value,
                        "candidate_value": candidate_value,
                        "candidate_minus_baseline": delta,
                        "favorable_change": delta if higher_is_better else -delta,
                    }
                )
    return pd.DataFrame(rows)


def plot_test_metrics(table, baseline_label, candidate_label, figure_dir):
    groups = [(target, architecture) for target in TARGETS for architecture in ARCHITECTURES]
    x = np.arange(len(groups), dtype=float)
    labels = [
        f"{TARGET_LABELS[target]}\n{ARCHITECTURE_LABELS[architecture]}"
        for target, architecture in groups
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for axis, (metric, metric_label, _) in zip(axes.flat, METRICS):
        selected = table.loc[table["metric"].eq(metric)].set_index(["target", "architecture"])
        baseline_values = np.array(
            [selected.loc[group, "baseline_value"] for group in groups], dtype=float
        )
        candidate_values = np.array(
            [selected.loc[group, "candidate_value"] for group in groups], dtype=float
        )
        width = 0.36
        first = axis.bar(x - width / 2, baseline_values, width, color=COLORS[0], label=baseline_label)
        second = axis.bar(x + width / 2, candidate_values, width, color=COLORS[1], label=candidate_label)
        axis.bar_label(first, fmt="%.3f", fontsize=7, padding=2)
        axis.bar_label(second, fmt="%.3f", fontsize=7, padding=2)
        axis.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_xticks(x, labels)
        axis.set_ylabel(metric_label)
        axis.set_title(f"Test {metric_label}")
        axis.grid(axis="y", alpha=0.22)
        axis.legend()
        values = np.concatenate([baseline_values, candidate_values])
        padding = max(float(np.ptp(values)) * 0.25, 0.08)
        axis.set_ylim(min(0, float(values.min()) - padding), float(values.max()) + padding)
    figure.suptitle("Controlled 20-frame regression comparison: test performance", fontsize=15)
    figure.tight_layout()
    figure.savefig(figure_dir / "test_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_favorable_change(table, candidate_label, figure_dir):
    row_groups = [(target, architecture) for target in TARGETS for architecture in ARCHITECTURES]
    values = np.array(
        [
            [
                table.loc[
                    table["target"].eq(target)
                    & table["architecture"].eq(architecture)
                    & table["metric"].eq(metric),
                    "favorable_change",
                ].iloc[0]
                for metric, _, _ in METRICS
            ]
            for target, architecture in row_groups
        ],
        dtype=float,
    )
    limit = max(float(np.abs(values).max()), 1e-6)
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(METRICS)), [label for _, label, _ in METRICS])
    axis.set_yticks(
        np.arange(len(row_groups)),
        [
            f"{TARGET_LABELS[target]} | {ARCHITECTURE_LABELS[architecture]}"
            for target, architecture in row_groups
        ],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:+.3f}",
                ha="center",
                va="center",
                color="white" if abs(values[row, column]) > 0.52 * limit else "#222222",
                fontsize=9,
            )
    axis.set_title(f"{candidate_label}: favorable change from baseline")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Positive = candidate is better; negative = candidate is worse")
    figure.tight_layout()
    figure.savefig(figure_dir / "favorable_change_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_predictions(aligned_predictions, baseline_label, candidate_label, figure_dir):
    figure, axes = plt.subplots(len(TARGETS), 2 * len(ARCHITECTURES), figsize=(17, 8.5), squeeze=False)
    variants = (("baseline", baseline_label, COLORS[0]), ("candidate", candidate_label, COLORS[1]))
    for row, target in enumerate(TARGETS):
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            aligned = aligned_predictions[(architecture, target)]
            y_true = aligned["y_true_baseline"].to_numpy(dtype=float)
            all_predictions = np.concatenate(
                [
                    aligned["y_pred_baseline"].to_numpy(dtype=float),
                    aligned["y_pred_candidate"].to_numpy(dtype=float),
                ]
            )
            lower = float(min(y_true.min(), all_predictions.min()))
            upper = float(max(y_true.max(), all_predictions.max()))
            padding = max((upper - lower) * 0.06, 0.05)
            limits = (lower - padding, upper + padding)
            for variant_index, (suffix, variant_label, color) in enumerate(variants):
                axis = axes[row, architecture_index * 2 + variant_index]
                y_pred = aligned[f"y_pred_{suffix}"].to_numpy(dtype=float)
                axis.scatter(y_true, y_pred, s=19, alpha=0.62, color=color, edgecolors="none")
                axis.plot(limits, limits, color="#333333", linestyle="--", linewidth=1)
                axis.axhline(0, color="#888888", linewidth=0.7)
                axis.axvline(0, color="#888888", linewidth=0.7)
                axis.set_xlim(limits)
                axis.set_ylim(limits)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("True abnormal score")
                axis.set_ylabel("Predicted abnormal score")
                axis.set_title(
                    f"{TARGET_LABELS[target]} | {ARCHITECTURE_LABELS[architecture]}\n"
                    f"{variant_label} (n={len(aligned)})"
                )
                axis.grid(alpha=0.17)
    figure.suptitle("Paired video-level test predictions", fontsize=15, y=1.002)
    figure.tight_layout()
    figure.savefig(figure_dir / "test_prediction_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_validation_history(baseline_dir, candidate_dir, baseline_label, candidate_label, figure_dir):
    histories = {
        baseline_label: pd.read_csv(baseline_dir / "history_all.csv"),
        candidate_label: pd.read_csv(candidate_dir / "history_all.csv"),
    }
    figure, axes = plt.subplots(len(TARGETS), len(ARCHITECTURES), figsize=(15, 8.5), squeeze=False)
    for row, target in enumerate(TARGETS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            score_axis = axis.twinx()
            for color, (variant_label, history) in zip(COLORS, histories.items()):
                selected = history.loc[
                    history["architecture"].eq(architecture) & history["target"].eq(target)
                ].sort_values("global_epoch")
                if selected.empty:
                    raise RuntimeError(f"Missing history for {architecture}/{target} in {variant_label}")
                epochs = selected["global_epoch"].to_numpy()
                axis.plot(epochs, selected["val_loss"], color=color, linewidth=1.4)
                score_axis.plot(
                    epochs,
                    selected["val_pearson_r"],
                    color=color,
                    linewidth=1.15,
                    linestyle="--",
                )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Validation SmoothL1 loss")
            score_axis.set_ylabel("Validation Pearson r")
            score_axis.axhline(0, color="#777777", linewidth=0.7, linestyle=":")
            axis.set_title(f"{TARGET_LABELS[target]} | {ARCHITECTURE_LABELS[architecture]}")
            axis.grid(axis="y", alpha=0.2)
    legend = [
        Line2D([0], [0], color=COLORS[0], lw=1.5, label=baseline_label),
        Line2D([0], [0], color=COLORS[1], lw=1.5, label=candidate_label),
        Line2D([0], [0], color="#444444", lw=1.5, label="Validation loss"),
        Line2D([0], [0], color="#444444", lw=1.2, linestyle="--", label="Validation Pearson r"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("Validation trajectories on matched data splits", fontsize=15, y=0.995)
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(figure_dir / "validation_history_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    style_plots()

    baseline_metrics = load_test_metrics(baseline_dir)
    candidate_metrics = load_test_metrics(candidate_dir)
    metric_table = build_metric_table(
        baseline_metrics, candidate_metrics, args.baseline_label, args.candidate_label
    )
    metric_table.to_csv(table_dir / "test_metric_comparison.csv", index=False)

    aligned_predictions = {}
    alignment_rows = []
    for architecture in ARCHITECTURES:
        for target in TARGETS:
            aligned = load_aligned_test_predictions(
                baseline_dir, candidate_dir, architecture, target
            )
            aligned_predictions[(architecture, target)] = aligned
            alignment_rows.append(
                {
                    "architecture": architecture,
                    "target": target,
                    "matched_test_videos": len(aligned),
                    "identical_video_set": True,
                    "identical_targets": True,
                    "frames_per_video_baseline": int(aligned["frame_count_baseline"].iloc[0]),
                    "frames_per_video_candidate": int(aligned["frame_count_candidate"].iloc[0]),
                }
            )
    pd.DataFrame(alignment_rows).to_csv(table_dir / "data_alignment.csv", index=False)

    plot_test_metrics(metric_table, args.baseline_label, args.candidate_label, figure_dir)
    plot_favorable_change(metric_table, args.candidate_label, figure_dir)
    plot_predictions(aligned_predictions, args.baseline_label, args.candidate_label, figure_dir)
    plot_validation_history(
        baseline_dir,
        candidate_dir,
        args.baseline_label,
        args.candidate_label,
        figure_dir,
    )
    print(f"Saved 4 comparison figures and 2 audit tables to {output_dir}")


if __name__ == "__main__":
    main()
