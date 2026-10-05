"""Compare Exp6 time windows on full and exactly shared held-out pairs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGET_UNITS
from .plot_results import DISPLAY


METRICS = (
    ("mae", "MAE", "mae"),
    ("pearson_r", "Pearson r", "pearson_r"),
    ("r2", "R2", "r2"),
    ("direction_balanced_accuracy", "Direction bACC", "direction_bacc"),
)


def _predictions(root, target):
    path = root / "runs" / target / "pair_predictions.csv"
    frame = pd.read_csv(path, dtype={"pair_id": str, "hospital_id": str})
    frame = frame.loc[frame.split.eq("test")].set_index("pair_id")
    if frame.index.has_duplicates or not frame.frame_count.eq(20).all():
        raise AssertionError(f"Invalid held-out pair predictions: {path}")
    return frame


def _metrics(frame):
    n = len(frame)
    result = {"n_pairs": n, "n_patients": frame.hospital_id.nunique()}
    if not n:
        return {**result, **{field: np.nan for field, _, _ in METRICS}}
    truth = frame.y_true.to_numpy(float)
    predicted = frame.y_pred.to_numpy(float)
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise AssertionError("Non-finite held-out pair predictions")
    result["mae"] = float(mean_absolute_error(truth, predicted))
    result["r2"] = float(r2_score(truth, predicted)) if n >= 2 and np.std(truth) > 0 else np.nan
    result["pearson_r"] = (
        float(np.corrcoef(truth, predicted)[0, 1])
        if n >= 2 and np.std(truth) > 0 and np.std(predicted) > 0 else np.nan
    )
    nonzero = ~np.isclose(truth, 0, atol=1e-12)
    direction = truth[nonzero] > 0
    output = predicted[nonzero]
    result["direction_n"] = int(nonzero.sum())
    result["direction_balanced_accuracy"] = (
        float(balanced_accuracy_score(direction, output > 0))
        if set(direction) == {False, True} else np.nan
    )
    result["direction_roc_auc"] = (
        float(roc_auc_score(direction, output))
        if set(direction) == {False, True} else np.nan
    )
    return result


def _plot(table, targets, hours, cohort, metric, label, stem, figures):
    rows, columns = target_grid_shape(len(targets))
    figure, axes = plt.subplots(rows, columns,
                                figsize=target_grid_figsize(rows, columns), squeeze=False)
    indexed = table.set_index("target")
    for axis, target in zip(axes.flat, targets):
        row = indexed.loc[target]
        values = [row[f"baseline_{metric}"], row[f"candidate_{metric}"]]
        axis.bar((0, 1), values, color=("#677888", "#287F78"), width=0.62)
        axis.set_xticks((0, 1), ("24 h", f"{hours} h"))
        axis.set_title(f"{DISPLAY.get(target, target)} | "
                       f"n={int(row['baseline_n_pairs'])}/{int(row['candidate_n_pairs'])}")
        unit = TARGET_UNITS[target] if metric == "mae" else ""
        axis.set_ylabel(f"{label} ({unit})" if unit else label)
        axis.grid(axis="y", alpha=0.2)
        if metric == "direction_balanced_accuracy":
            axis.set_ylim(0, 1.05)
        elif metric == "pearson_r":
            axis.set_ylim(-1.05, 1.05)
        elif metric == "mae":
            axis.set_ylim(bottom=0)
        for x, value in enumerate(values):
            if np.isfinite(value):
                axis.annotate(f"{value:.2f}", (x, value),
                              xytext=(0, 4 if value >= 0 else -4),
                              textcoords="offset points", ha="center",
                              va="bottom" if value >= 0 else "top", fontsize=8)
    figure.suptitle(f"Exp6 {cohort} held-out pairs | {label}", fontsize=14)
    figure.tight_layout()
    figure.savefig(figures / f"{cohort}_test_{stem}.png", dpi=180)
    plt.close(figure)


def plot_comparison(baseline_dir, candidate_dir, targets, hours):
    baseline_dir, candidate_dir = Path(baseline_dir), Path(candidate_dir)
    targets = tuple(targets)
    for root in (baseline_dir, candidate_dir):
        index = pd.read_csv(root / "run_index.csv")
        if len(index) != len(targets) or set(index.target) != set(targets) or not index.status.eq("ok").all():
            raise RuntimeError(f"Incomplete comparison source: {root}")
    tables = {"full": [], "shared": []}
    for target in targets:
        baseline = _predictions(baseline_dir, target)
        candidate = _predictions(candidate_dir, target)
        common = baseline.index.intersection(candidate.index)
        for column in ("hospital_id", "first_video_id", "second_video_id"):
            if not baseline.loc[common, column].equals(candidate.loc[common, column]):
                raise AssertionError(f"Shared {column} differs: {target}")
        if not np.allclose(baseline.loc[common, "raw_delta"],
                           candidate.loc[common, "raw_delta"], rtol=0, atol=1e-10):
            raise AssertionError(f"Shared laboratory changes differ: {target}")
        for cohort, first, second in (
            ("full", baseline, candidate),
            ("shared", baseline.loc[common], candidate.loc[common]),
        ):
            tables[cohort].append({
                "target": target,
                **{f"baseline_{key}": value for key, value in _metrics(first).items()},
                **{f"candidate_{key}": value for key, value in _metrics(second).items()},
            })
    figures = candidate_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for cohort, rows in tables.items():
        table = pd.DataFrame(rows)
        table.to_csv(candidate_dir / f"{cohort}_test_comparison.csv", index=False)
        for field, label, stem in METRICS:
            _plot(table, targets, hours, cohort, field, label, stem, figures)
    print(f"[window-comparison-complete] hours={hours} figures={figures}", flush=True)
