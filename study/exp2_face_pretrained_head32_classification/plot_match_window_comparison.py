"""Compare binary matching windows on full and exactly shared test cohorts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp2_binary_classification_common.plot_results import DISPLAY


def _predictions(root, target):
    path = root / "runs" / "efficientnet_b0" / target / "video_predictions.csv"
    frame = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
    frame = frame.loc[frame["split"].eq("test")].set_index("video_id")
    if frame.index.has_duplicates or not frame.input_count.eq(20).all():
        raise AssertionError(f"Invalid test predictions: {path}")
    return frame


def _binary_metrics(frame):
    if len(frame) == 0:
        return {"n_videos": 0, "n_patients": 0, "n_positive": 0,
                "balanced_accuracy": np.nan, "roc_auc": np.nan}
    labels = frame.y_true.to_numpy(int)
    probabilities = frame.y_probability.to_numpy(float)
    both_classes = set(labels) == {0, 1}
    return {
        "n_videos": len(frame), "n_patients": frame.hospital_id.nunique(),
        "n_positive": int(labels.sum()),
        "balanced_accuracy": (
            balanced_accuracy_score(labels, probabilities >= 0.5)
            if both_classes else np.nan
        ),
        "roc_auc": roc_auc_score(labels, probabilities) if both_classes else np.nan,
    }


def _plot(table, targets, hours, metric, cohort, destination):
    rows, columns = target_grid_shape(len(targets))
    figure, axes = plt.subplots(rows, columns,
                                figsize=target_grid_figsize(rows, columns), squeeze=False)
    indexed = table.set_index("target")
    for axis, target in zip(axes.flat, targets):
        row = indexed.loc[target]
        values = [row[f"reference_{metric}"], row[f"candidate_{metric}"]]
        bars = axis.bar((0, 1), values, color=("#5C6975", "#278078"), width=0.62)
        for bar, value in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2
            if np.isfinite(value):
                axis.annotate(f"{value:.3f}", (x, value), xytext=(0, 3),
                              textcoords="offset points", ha="center", fontsize=8)
            else:
                axis.text(x, 0.05, "N/A", ha="center", fontsize=8)
        axis.set_title(
            f"{DISPLAY.get(target, target)}\n"
            f"n={int(row['reference_n_videos'])}/{int(row['candidate_n_videos'])}",
            fontsize=10,
        )
        axis.set_xticks((0, 1), ("24 h", f"{hours} h"))
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("bACC" if metric == "balanced_accuracy" else "AUROC")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"Classification | {cohort} test videos", fontsize=14)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_comparison(reference_dir, candidate_dir, targets, hours):
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    targets = tuple(targets)
    run_index = pd.read_csv(candidate_dir / "run_index.csv")
    if len(run_index) != len(targets) or not run_index.status.eq("complete").all():
        raise RuntimeError("Cannot compare an incomplete classification ablation")
    tables = {"full": [], "shared": []}
    for target in targets:
        baseline = _predictions(reference_dir, target)
        candidate = _predictions(candidate_dir, target)
        common = baseline.index.intersection(candidate.index)
        if not baseline.loc[common, ["hospital_id", "y_true"]].equals(
            candidate.loc[common, ["hospital_id", "y_true"]]
        ):
            raise AssertionError(f"Shared held-out labels differ for {target}")
        for cohort, left, right in (
            ("full", baseline, candidate),
            ("shared", baseline.loc[common], candidate.loc[common]),
        ):
            left_metrics = _binary_metrics(left)
            right_metrics = _binary_metrics(right)
            tables[cohort].append({
                "target": target,
                **{f"reference_{key}": value for key, value in left_metrics.items()},
                **{f"candidate_{key}": value for key, value in right_metrics.items()},
            })
    figures = candidate_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for cohort, rows in tables.items():
        table = pd.DataFrame(rows)
        table.to_csv(candidate_dir / f"{cohort}_test_comparison.csv", index=False)
        for metric, stem in (("balanced_accuracy", "bacc"), ("roc_auc", "auroc")):
            _plot(table, targets, hours, metric, cohort,
                  figures / f"{cohort}_test_{stem}.png")
    print(f"[comparison-complete] hours={hours} figures={figures}", flush=True)
