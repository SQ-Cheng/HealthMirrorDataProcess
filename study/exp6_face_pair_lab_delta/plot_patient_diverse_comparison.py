"""Compare the patient-diverse 30/40 variant with the completed Exp6 baseline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .plot_results import DISPLAY


def _test_metrics(output_dir, targets):
    frame = pd.read_csv(output_dir / "metrics_all.csv")
    test = frame.loc[frame.split.eq("test")].set_index("target")
    if len(test) != len(targets) or set(test.index) != set(targets):
        raise AssertionError(f"Missing or duplicate test targets: {output_dir}")
    return test.loc[list(targets)]


def _check_pairs(baseline_dir, variant_dir, target):
    paths = [root / "runs" / target / "pair_predictions.csv"
             for root in (baseline_dir, variant_dir)]
    frames = [pd.read_csv(path, dtype={"pair_id": str, "hospital_id": str})
              .sort_values("pair_id").reset_index(drop=True) for path in paths]
    first, second = frames
    keys = ("pair_id", "hospital_id", "split", "first_video_id", "second_video_id")
    if len(first) != len(second) or any(
        not first[key].astype(str).equals(second[key].astype(str)) for key in keys
    ):
        raise AssertionError(f"Comparison does not use identical pairs: {target}")
    if not np.allclose(first.raw_delta, second.raw_delta, rtol=0, atol=1e-10):
        raise AssertionError(f"Comparison labels differ: {target}")
    if not first.frame_count.eq(20).all() or not second.frame_count.eq(20).all():
        raise AssertionError(f"Comparison frame count differs: {target}")


def _plot_test_metric(table, targets, column, title, filename, figure_dir):
    rows, cols = target_grid_shape(len(targets))
    fig, axes = plt.subplots(rows, cols, figsize=target_grid_figsize(rows, cols), squeeze=False)
    for axis, target in zip(axes.flat, targets):
        values = [table.loc[target, f"baseline_{column}"],
                  table.loc[target, f"patient_diverse_30_40_{column}"]]
        axis.bar([0, 1], values, color=["#73808a", "#257f77"], width=0.62)
        axis.set_xticks([0, 1], ["Baseline", "Patient diverse\n30/40"])
        axis.set_title(DISPLAY.get(target, target))
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", labelsize=8)
        for x, value in enumerate(values):
            if np.isfinite(value):
                axis.annotate(f"{value:.3g}", (x, value), xytext=(0, 4),
                              textcoords="offset points", ha="center", fontsize=8)
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(figure_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(baseline_dir, variant_dir, targets):
    baseline_dir, variant_dir = Path(baseline_dir), Path(variant_dir)
    targets = tuple(targets)
    for root in (baseline_dir, variant_dir):
        index = pd.read_csv(root / "run_index.csv")
        if len(index) != len(targets) or set(index.target) != set(targets) or not index.status.eq("ok").all():
            raise AssertionError(f"Incomplete comparison runs: {root}")
    baseline = _test_metrics(baseline_dir, targets)
    variant = _test_metrics(variant_dir, targets)
    for target in targets:
        _check_pairs(baseline_dir, variant_dir, target)
    if not baseline.n.equals(variant.n):
        raise AssertionError("Comparison test sample counts differ")

    columns = ("mae", "r2", "pearson_r", "direction_balanced_accuracy")
    table = pd.DataFrame({
        "target": targets,
        "test_pairs": baseline.n.to_numpy(int),
    }).set_index("target")
    for column in columns:
        table[f"baseline_{column}"] = baseline[column]
        table[f"patient_diverse_30_40_{column}"] = variant[column]
        table[f"delta_{column}"] = variant[column] - baseline[column]
    table.reset_index().to_csv(variant_dir / "baseline_comparison.csv", index=False)

    figure_dir = variant_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, filename in (
        ("mae", "Exp6 test MAE (lower is better)", "baseline_vs_patient_diverse_test_mae.png"),
        ("pearson_r", "Exp6 test Pearson r (higher is better)", "baseline_vs_patient_diverse_test_r.png"),
        ("r2", "Exp6 test R2 (higher is better)", "baseline_vs_patient_diverse_test_r2.png"),
    ):
        _plot_test_metric(table, targets, metric, title, filename, figure_dir)

    rows, cols = target_grid_shape(len(targets))
    fig, axes = plt.subplots(rows, cols, figsize=target_grid_figsize(rows, cols), squeeze=False)
    for axis, target in zip(axes.flat, targets):
        for root, label, color in ((baseline_dir, "Baseline", "#73808a"),
                                   (variant_dir, "Patient diverse 30/40", "#257f77")):
            history = pd.read_csv(root / "runs" / target / "history.csv")
            axis.plot(history.global_epoch, history.val_mae, color=color, label=label)
            boundary = history.loc[history.stage.ne(history.stage.shift()), "global_epoch"].iloc[1:]
            if len(boundary):
                axis.axvline(boundary.iloc[0] - 0.5, color=color, linestyle=":", alpha=0.65)
        axis.set_title(DISPLAY.get(target, target))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation MAE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    fig.suptitle("Exp6 validation histories: baseline vs patient-diverse 30/40")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(figure_dir / "baseline_vs_patient_diverse_val_history.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[comparison-complete] directory={figure_dir}", flush=True)


if __name__ == "__main__":
    from .config import OUTPUT_DIR, TARGETS
    plot_comparison(OUTPUT_DIR, OUTPUT_DIR / "shared_patient_diverse_30_40", TARGETS)
