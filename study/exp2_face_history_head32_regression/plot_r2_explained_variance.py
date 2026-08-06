"""Recompute R2/explained variance and plot every completed history model."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, r2_score


EXP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXP_DIR / "outputs"
VARIANTS = {
    "20frame": "Unweighted",
    "20frame_range_weighted": "Range weighted",
}
ARCHITECTURES = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
TARGETS = {
    "hemoglobin_low": "Hemoglobin",
    "po2_low": "PO2",
    "oxyhemoglobin_fraction": "Oxyhemoglobin fraction",
}
COLORS = {"Unweighted": "#4C78A8", "Range weighted": "#E15759"}
FIGURE_TITLE = "Face + lab-history regression: video-level test goodness of fit"


def _scores(predictions, split):
    selected = predictions.loc[predictions["split"].eq(split)]
    truth = selected["y_true"].to_numpy(np.float64)
    estimate = selected["y_pred"].to_numpy(np.float64)
    if len(truth) < 2 or not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError(f"Invalid predictions for split={split}")
    return {
        "n": len(truth),
        "r2": float(r2_score(truth, estimate)),
        "explained_variance": float(explained_variance_score(truth, estimate)),
    }


def _update_variant(variant, label):
    output = OUTPUT_ROOT / variant
    run_index = pd.read_csv(output / "run_index.csv")
    if run_index.empty or not run_index["status"].eq("ok").all():
        raise RuntimeError(f"Incomplete run index: {output}")
    rows = []
    for run in run_index.itertuples(index=False):
        run_dir = Path(run.run_dir)
        predictions = pd.read_csv(run_dir / "video_predictions.csv")
        run_metrics = pd.read_csv(run_dir / "metrics.csv")
        for split in ("train", "val", "test"):
            scores = _scores(predictions, split)
            existing = run_metrics.loc[run_metrics["split"].eq(split)]
            if len(existing) != 1:
                raise RuntimeError(f"Missing metric row: {run.architecture}/{run.target}/{split}")
            old_r2 = float(existing.iloc[0]["r2"])
            # Prediction CSV round trips can introduce a few ULPs of drift.
            if not np.isclose(old_r2, scores["r2"], rtol=0, atol=1e-7):
                raise RuntimeError(
                    f"Stored R2 mismatch for {run.architecture}/{run.target}/{split}: "
                    f"stored={old_r2}, recomputed={scores['r2']}"
                )
            mask = run_metrics["split"].eq(split)
            run_metrics.loc[mask, "r2"] = scores["r2"]
            run_metrics.loc[mask, "explained_variance"] = scores["explained_variance"]
            rows.append({
                "variant": variant,
                "variant_label": label,
                "architecture": run.architecture,
                "target": run.target,
                "split": split,
                **scores,
            })
        run_metrics.to_csv(run_dir / "metrics.csv", index=False)
    summary = pd.DataFrame(rows)
    metrics_all = pd.read_csv(output / "metrics_all.csv")
    metrics_all = metrics_all.drop(columns=["explained_variance"], errors="ignore")
    metrics_all = metrics_all.merge(
        summary[["architecture", "target", "split", "explained_variance"]],
        on=["architecture", "target", "split"],
        how="left",
        validate="one_to_one",
    )
    if metrics_all["explained_variance"].isna().any():
        raise RuntimeError(f"Failed to update all aggregate rows in {output}")
    metrics_all.to_csv(output / "metrics_all.csv", index=False)
    return summary


def _plot(summary, path):
    test = summary.loc[summary["split"].eq("test")].copy()
    figure, axes = plt.subplots(len(TARGETS), 2, figsize=(15, 12), squeeze=False)
    for row_index, (target, target_label) in enumerate(TARGETS.items()):
        target_rows = test.loc[test["target"].eq(target)]
        for column_index, (metric, metric_label) in enumerate(
            (("r2", "R²"), ("explained_variance", "Explained variance"))
        ):
            axis = axes[row_index, column_index]
            positions, values, labels, colors = [], [], [], []
            for architecture, architecture_label in ARCHITECTURES.items():
                for variant_label in VARIANTS.values():
                    selected = target_rows.loc[
                        target_rows["architecture"].eq(architecture)
                        & target_rows["variant_label"].eq(variant_label)
                    ]
                    if selected.empty:
                        continue
                    positions.append(len(positions))
                    values.append(float(selected.iloc[0][metric]))
                    labels.append(f"{architecture_label}\n{variant_label}")
                    colors.append(COLORS[variant_label])
            if not values:
                axis.set_visible(False)
                continue
            bars = axis.bar(positions, values, color=colors, width=0.72)
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            axis.set_xticks(positions, labels, rotation=16, ha="right")
            axis.axhline(0, color="#555555", linewidth=0.8, linestyle=":")
            axis.set_ylabel(metric_label)
            axis.set_title(f"{target_label}: test {metric_label}")
            axis.grid(axis="y", alpha=0.2)
            lower, upper = min(values), max(values)
            padding = max((upper - lower) * 0.25, 0.08)
            axis.set_ylim(min(0, lower - padding), upper + padding)
    figure.suptitle(FIGURE_TITLE, fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    summaries = [
        _update_variant(variant, label)
        for variant, label in VARIANTS.items()
        if (OUTPUT_ROOT / variant / "run_index.csv").is_file()
    ]
    if not summaries:
        raise RuntimeError("No completed variants found")
    summary = pd.concat(summaries, ignore_index=True).sort_values(
        ["target", "architecture", "variant", "split"]
    )
    table_dir = OUTPUT_ROOT / "tables"
    figure_dir = OUTPUT_ROOT / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "r2_explained_variance.csv"
    figure_path = figure_dir / "r2_explained_variance.png"
    summary.to_csv(table_path, index=False)
    _plot(summary, figure_path)
    print(f"Saved {len(summary)} metric rows to {table_path}")
    print(f"Saved comparison figure to {figure_path}")


if __name__ == "__main__":
    main()
