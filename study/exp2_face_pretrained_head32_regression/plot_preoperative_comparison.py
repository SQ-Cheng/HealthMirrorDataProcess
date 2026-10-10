"""Compare full cohorts and the common test subset with unchanged labels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape, target_grid_figsize
from .config import ALL_REGRESSION_TARGETS, SCORE_DEFINITIONS
from .plot_results import TASK_LABELS, TASK_UNITS
from .train import _regression_metrics


def plot_comparison(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    tables = candidate / "tables"; tables.mkdir(exist_ok=True)
    figures = candidate / "figures"; figures.mkdir(exist_ok=True)
    full, shared, alignment = [], [], []
    for target in ALL_REGRESSION_TARGETS:
        predictions = []
        for name, root in (("Nearest 24h", reference), ("Preoperative unrestricted", candidate)):
            metrics = pd.read_csv(root / "metrics_all.csv")
            selected = metrics.loc[metrics.target.eq(target) & metrics.split.eq("test")]
            if len(selected) != 1:
                raise RuntimeError(f"Missing or duplicate metrics: {root}/{target}")
            full.append({**selected.iloc[0].to_dict(), "protocol": name})
            path = root / f"runs/efficientnet_b0/{target}/video_predictions.csv"
            values = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
            predictions.append(values.loc[values.split.eq("test")])
        joined = predictions[0].merge(predictions[1], on=["hospital_id", "video_id"],
                                      suffixes=("_reference", "_candidate"), validate="one_to_one")
        unchanged = np.isclose(joined.y_true_reference, joined.y_true_candidate, rtol=0, atol=1e-7)
        alignment.append({"target": target, "reference_test_videos": len(predictions[0]),
                          "candidate_test_videos": len(predictions[1]), "shared_test_videos": len(joined),
                          "shared_test_identical_value_videos": int(unchanged.sum()),
                          "shared_test_changed_value_videos": int((~unchanged).sum())})
        common = joined.loc[unchanged]
        for name, suffix in (("Nearest 24h", "reference"), ("Preoperative unrestricted", "candidate")):
            metrics = _regression_metrics(common[f"y_true_{suffix}"], common[f"y_pred_{suffix}"],
                                          common[f"score_threshold_{suffix}"], SCORE_DEFINITIONS[target]["direction"])
            shared.append({"target": target, "protocol": name, **metrics})
        joined.assign(target=target, unchanged_value=unchanged).to_csv(tables / f"{target}_shared_test_predictions.csv", index=False)
    full = pd.DataFrame(full); shared = pd.DataFrame(shared)
    full.to_csv(tables / "full_cohort_test_comparison.csv", index=False)
    shared.to_csv(tables / "shared_test_unchanged_label_comparison.csv", index=False)
    pd.DataFrame(alignment).to_csv(tables / "test_cohort_alignment.csv", index=False)
    colors = ("#2878B5", "#CB6547")
    labels = ("Nearest 24h", "Preoperative unrestricted")
    x = np.arange(len(ALL_REGRESSION_TARGETS))
    figure, axes = plt.subplots(1, 2, figsize=(18, 5.5))
    for axis, metric in zip(axes, ("pearson_r", "r2")):
        for offset, (label, color) in enumerate(zip(labels, colors)):
            values = full.loc[full.protocol.eq(label)].set_index("target").loc[list(ALL_REGRESSION_TARGETS)]
            axis.bar(x + (offset - .5) * .36, values[metric], .36, label=label, color=color)
        axis.set_xticks(x, [TASK_LABELS[target] for target in ALL_REGRESSION_TARGETS], rotation=35, ha="right")
        axis.set_ylabel("Pearson r" if metric == "pearson_r" else "R2")
        axis.axhline(0, color="#666666", lw=.8)
        axis.grid(axis="y", alpha=.2); axis.legend()
    figure.suptitle("Full held-out test cohorts (sample sets and preoperative labels differ)")
    figure.tight_layout()
    figure.savefig(figures / "preoperative_vs_24h_test_correlation.png", dpi=180)
    plt.close(figure)
    rows, columns = target_grid_shape(len(ALL_REGRESSION_TARGETS))
    figure, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False)
    for axis, target in zip(axes.flat, ALL_REGRESSION_TARGETS):
        selected = shared.loc[shared.target.eq(target)].set_index("protocol")
        for offset, (label, color) in enumerate(zip(labels, colors)):
            axis.bar(np.arange(2) + (offset - .5) * .36,
                     selected.loc[label, ["mae", "rmse"]].to_numpy(float), .36, color=color, label=label)
        axis.set_xticks([0, 1], ["MAE", "RMSE"])
        axis.set(title=f"{TASK_LABELS[target]} | n={int(selected.iloc[0]['n'])}", ylabel=TASK_UNITS[target])
        axis.grid(axis="y", alpha=.2)
    for axis in axes.flat[len(ALL_REGRESSION_TARGETS):]: axis.axis("off")
    handles, legends = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legends, loc="lower center", ncol=2)
    figure.suptitle("Common held-out test videos with identical target values")
    figure.tight_layout(rect=(0, .05, 1, .96))
    figure.savefig(figures / "preoperative_vs_24h_shared_test_errors.png", dpi=180)
    plt.close(figure)
    print(f"[comparison] full-cohort and identical-label shared-test figures: {figures}", flush=True)
