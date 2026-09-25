"""Compare completed Exp2 face-only training and backbone ablations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGETS
from .plot_results import TASK_LABELS


LABELS = {
    "baseline": "EfficientNet | two-stage full",
    "one_stage_full": "EfficientNet | direct",
    "two_stage_tail30": "EfficientNet | tail 28%",
    "shufflenet_two_stage": "ShuffleNetV2 | two-stage",
}
COLORS = {
    "baseline": "#2878B5",
    "one_stage_full": "#D05B3E",
    "two_stage_tail30": "#278245",
    "shufflenet_two_stage": "#8C4B99",
}


def _test_metrics(root, variant, architecture):
    root = Path(root)
    index = pd.read_csv(root / "run_index.csv")
    if len(index) != len(TARGETS) or not index.status.eq("ok").all():
        raise RuntimeError(f"Incomplete ablation: {root}")
    if set(index.target) != set(TARGETS) or set(index.architecture) != {architecture}:
        raise AssertionError(f"Wrong task or architecture inventory: {root}")
    if variant != "baseline" and not (root / "COMPLETE").is_file():
        raise RuntimeError(f"Missing completion marker: {root}")
    frame = pd.read_csv(root / "metrics_all.csv")
    frame = frame.loc[frame.split.eq("test")].copy()
    if len(frame) != len(TARGETS) or set(frame.target) != set(TARGETS):
        raise AssertionError(f"Incomplete test metrics: {root}")
    frame["variant"] = variant
    frame["model_label"] = LABELS[variant]
    return frame


def _plot_group_effects(test, figure_dir):
    baseline = test.loc[test.variant.eq("baseline")].set_index("target")
    variants = ("one_stage_full", "two_stage_tail30", "shufflenet_two_stage")
    x = np.arange(len(TARGETS))
    figure, axes = plt.subplots(3, 2, figsize=(17, 13), squeeze=False)
    for row, variant in enumerate(variants):
        selected = test.loc[test.variant.eq(variant)].set_index("target")
        relative_mae = 100 * (
            selected.loc[list(TARGETS), "mae"].to_numpy(float)
            / baseline.loc[list(TARGETS), "mae"].to_numpy(float) - 1
        )
        delta_r = (
            selected.loc[list(TARGETS), "pearson_r"].to_numpy(float)
            - baseline.loc[list(TARGETS), "pearson_r"].to_numpy(float)
        )
        for axis, values, ylabel in (
            (axes[row, 0], relative_mae, "Test MAE change vs baseline (%)"),
            (axes[row, 1], delta_r, "Test Pearson r change vs baseline"),
        ):
            improved = values < 0 if ylabel.startswith("Test MAE") else values > 0
            axis.bar(x, values, color=np.where(improved, "#278245", "#D05B3E"))
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.set_xticks(x, [TASK_LABELS[target] for target in TARGETS],
                            rotation=24, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(LABELS[variant])
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Face-only regression ablations: held-out test effects")
    figure.tight_layout()
    figure.savefig(figure_dir / "group_effects.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_per_target(test, figure_dir):
    variants = tuple(LABELS)
    baseline = test.loc[test.variant.eq("baseline")].set_index("target")
    rows, columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False,
    )
    for axis, target in zip(axes.flat, TARGETS):
        subset = test.loc[test.target.eq(target)].set_index("variant").loc[list(variants)]
        x = np.arange(len(variants))
        ratios = subset.mae.to_numpy(float) / float(baseline.loc[target, "mae"])
        axis.bar(x, ratios, color=[COLORS[name] for name in variants], alpha=0.82,
                 label="MAE / baseline MAE")
        axis.axhline(1, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Relative test MAE")
        axis.set_xticks(x, ("Baseline", "Direct", "Tail 28%", "ShuffleNet"),
                        rotation=20, ha="right")
        axis.set_title(TASK_LABELS[target])
        axis.grid(axis="y", alpha=0.18)
        r_axis = axis.twinx()
        r_values = subset.pearson_r.to_numpy(float)
        r_axis.plot(x, r_values, "o-", color="#222222",
                    label="Test Pearson r")
        r_axis.set_ylabel("Pearson r")
        finite = r_values[np.isfinite(r_values)]
        if len(finite):
            low, high = float(finite.min()), float(finite.max())
            pad = max((high - low) * 0.15, 0.05)
            r_axis.set_ylim(max(-1.0, low - pad), min(1.0, high + pad))
    figure.suptitle("All four face-only regressors on identical held-out videos")
    figure.tight_layout()
    figure.savefig(figure_dir / "per_target_models.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(base_dir, ablation_dir, variants):
    base_dir, ablation_dir = Path(base_dir), Path(ablation_dir)
    frames = [_test_metrics(base_dir, "baseline", "efficientnet_b0")]
    baseline_counts = frames[0].set_index("target")["n"].sort_index()
    for name, architecture, _ in variants:
        frame = _test_metrics(ablation_dir / name, name, architecture)
        pd.testing.assert_series_equal(
            frame.set_index("target")["n"].sort_index(), baseline_counts,
            check_names=False,
        )
        frames.append(frame)
    test = pd.concat(frames, ignore_index=True)
    output_dir = ablation_dir / "comparison"
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(output_dir / "all_test_metrics.csv", index=False)
    _plot_group_effects(test, figure_dir)
    _plot_per_target(test, figure_dir)
    print(f"[comparison-complete] output={output_dir}", flush=True)


if __name__ == "__main__":
    from .run_ablations import ABLATION_DIR, BASE_DIR, VARIANTS
    plot_comparison(BASE_DIR, ABLATION_DIR, VARIANTS)
