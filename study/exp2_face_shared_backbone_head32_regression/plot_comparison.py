"""Reference comparisons explicitly distinguish changed splits and common tests."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape, target_grid_figsize
from .config import ALL_REGRESSION_TARGETS, SCORE_DEFINITIONS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from study.exp2_face_pretrained_head32_regression.train import _regression_metrics


def plot_comparison(reference, candidate, names=("Independent backbone", "Shared backbone"),
                    experiment_label="Shared-backbone ablation", figure_prefix="shared_vs_independent"):
    reference, candidate = Path(reference), Path(candidate)
    figures = candidate / "figures"; figures.mkdir(exist_ok=True)
    tables = candidate / "tables"; tables.mkdir(exist_ok=True)
    colors = ("#2878B5", "#CB6547")
    full, common, audits = [], [], []
    metrics = [pd.read_csv(root / "metrics_all.csv") for root in (reference, candidate)]
    for target in ALL_REGRESSION_TARGETS:
        predictions = []
        for name, root, values in zip(names, (reference, candidate), metrics):
            row = values.loc[values.target.eq(target) & values.split.eq("test")]
            if len(row) != 1: raise RuntimeError(f"Incomplete metrics: {root}/{target}")
            full.append({**row.iloc[0].to_dict(), "model": name})
            frame = pd.read_csv(root / f"runs/efficientnet_b0/{target}/video_predictions.csv",
                                dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
            predictions.append(frame.loc[frame.split.eq("test")])
        shared = predictions[0].merge(predictions[1], on=["hospital_id", "video_id"],
                                       suffixes=("_independent", "_shared"), validate="one_to_one")
        np.testing.assert_allclose(shared.y_true_independent, shared.y_true_shared, rtol=0, atol=1e-7)
        assert shared.frame_count_independent.eq(20).all() and shared.frame_count_shared.eq(20).all()
        for name, suffix in zip(names, ("independent", "shared")):
            row = _regression_metrics(shared[f"y_true_{suffix}"], shared[f"y_pred_{suffix}"],
                                       shared[f"score_threshold_{suffix}"], SCORE_DEFINITIONS[target]["direction"])
            common.append({"target": target, "model": name, **row})
        audits.append({"target": target, "independent_test_videos": len(predictions[0]),
                       "shared_test_videos": len(predictions[1]), "common_test_videos": len(shared),
                       "common_test_patients": shared.hospital_id.nunique(), "training_splits_identical": False})
        shared.assign(target=target).to_csv(tables / f"{target}_common_test_predictions.csv", index=False)
    full, common = pd.DataFrame(full), pd.DataFrame(common)
    full.to_csv(tables / "full_cohort_comparison.csv", index=False)
    common.to_csv(tables / "common_test_comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(tables / "comparison_cohort_audit.csv", index=False)
    x = np.arange(len(ALL_REGRESSION_TARGETS))
    figure, axes = plt.subplots(2, 2, figsize=(18, 11))
    for row, (values, cohort) in enumerate(((full, "Full test cohorts"), (common, "Common held-out test videos"))):
        for axis, metric, title in zip(axes[row], ("pearson_r", "r2"), ("Pearson r", "R2")):
            for position, (name, color) in enumerate(zip(names, colors)):
                selected = values.loc[values.model.eq(name)].set_index("target").loc[list(ALL_REGRESSION_TARGETS)]
                axis.bar(x + (position - .5) * .36, selected[metric], .36, color=color, label=name)
            axis.set_xticks(x, [TASK_LABELS[t] for t in ALL_REGRESSION_TARGETS], rotation=30, ha="right")
            axis.set(title=f"{cohort}: {title}", ylabel=title)
            axis.axhline(0, color="#666666", lw=.8); axis.grid(axis="y", alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle(f"{experiment_label} | Different patient splits; no matched-split independent control")
    figure.tight_layout(rect=(0, .04, 1, .96))
    figure.savefig(figures / f"{figure_prefix}_correlations.png", dpi=180)
    plt.close(figure)
    rows, columns = target_grid_shape(len(ALL_REGRESSION_TARGETS))
    figure, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False)
    for axis, target in zip(axes.flat, ALL_REGRESSION_TARGETS):
        values = common.loc[common.target.eq(target)].set_index("model")
        for position, (name, color) in enumerate(zip(names, colors)):
            axis.bar(np.arange(2) + (position - .5) * .36, values.loc[name, ["mae", "rmse"]].to_numpy(float),
                     .36, label=name, color=color)
        axis.set_xticks([0, 1], ["MAE", "RMSE"])
        axis.set(title=f"{TASK_LABELS[target]} | n={int(values.iloc[0]['n'])}", ylabel=TASK_UNITS[target])
        axis.grid(axis="y", alpha=.2)
    for axis in axes.flat[len(ALL_REGRESSION_TARGETS):]: axis.axis("off")
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Common held-out test video errors | Training patient assignments differ")
    figure.tight_layout(rect=(0, .05, 1, .96))
    figure.savefig(figures / f"{figure_prefix}_common_test_errors.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False)
    for axis, target in zip(axes.flat, ALL_REGRESSION_TARGETS):
        prediction = pd.read_csv(candidate / f"runs/efficientnet_b0/{target}/video_predictions.csv")
        prediction = prediction.loc[prediction.split.eq("test")]
        truth, estimates = prediction.y_true.to_numpy(float), prediction.y_pred.to_numpy(float)
        axis.scatter(truth, estimates, s=18, alpha=.65, color=colors[1], edgecolors="none")
        lower, upper = min(truth.min(), estimates.min()), max(truth.max(), estimates.max())
        pad = max((upper - lower) * .05, .01); limits = (lower - pad, upper + pad)
        axis.plot(limits, limits, "--", color="#666666", lw=1, label="Identity")
        if len(truth) > 1 and np.std(truth) > 0:
            slope, intercept = np.polyfit(truth, estimates, 1)
            axis.plot(limits, slope * np.asarray(limits) + intercept, color=colors[0], label="Linear fit")
        axis.set(xlabel=f"Measured ({TASK_UNITS[target]})", ylabel=f"Predicted ({TASK_UNITS[target]})",
                 title=f"{TASK_LABELS[target]} | n={len(truth)}", xlim=limits, ylim=limits)
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    for axis in axes.flat[len(ALL_REGRESSION_TARGETS):]: axis.axis("off")
    figure.suptitle(f"{experiment_label}: held-out test predictions")
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(figures / "test_predicted_vs_true.png", dpi=180)
    plt.close(figure)
    print(f"[comparison-complete] {figures}", flush=True)
