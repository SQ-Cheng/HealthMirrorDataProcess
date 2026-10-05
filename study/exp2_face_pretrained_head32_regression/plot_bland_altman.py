"""Plot video-level test Bland-Altman analyses for a regression run."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .plot_results import TASKS, TASK_LABELS, TASK_UNITS, _style


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/ablations/lab_match_6h_face224"


def plot(output_dir):
    output_dir = Path(output_dir).resolve()
    index = pd.read_csv(output_dir / "run_index.csv")
    if (len(index) != len(TASKS) or set(index.target) != set(TASKS)
            or not index.status.eq("ok").all()
            or not index.architecture.eq("efficientnet_b0").all()):
        raise RuntimeError(f"Incomplete EfficientNet-B0 result: {output_dir}")
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    test_metrics = metrics.loc[metrics.split.eq("test")].set_index("target")
    if set(test_metrics.index) != set(TASKS) or len(test_metrics) != len(TASKS):
        raise RuntimeError("Incomplete video-level test metrics")

    _style()
    rows, columns = target_grid_shape(len(TASKS))
    figure, axes = plt.subplots(rows, columns,
                                figsize=target_grid_figsize(rows, columns), squeeze=False)
    summary = []
    for axis, target in zip(axes.flat, TASKS):
        path = output_dir / "runs" / "efficientnet_b0" / target / "video_predictions.csv"
        predictions = pd.read_csv(path, dtype={"video_id": str})
        predictions = predictions.loc[predictions.split.eq("test")]
        if (predictions.empty or predictions.video_id.duplicated().any()
                or not predictions.frame_count.eq(20).all()
                or len(predictions) != int(test_metrics.loc[target, "n"])):
            raise AssertionError(f"Invalid test video coverage: {target}")
        truth = predictions.y_true.to_numpy(float)
        predicted = predictions.y_pred.to_numpy(float)
        difference = predicted - truth
        if (not np.isfinite(truth).all() or not np.isfinite(predicted).all()
                or not np.allclose(difference, predictions.residual, rtol=0, atol=1e-5)):
            raise AssertionError(f"Test prediction values are inconsistent: {target}")
        mean = (predicted + truth) / 2
        bias = float(np.mean(difference))
        sd = float(np.std(difference, ddof=1))
        lower, upper = bias - 1.96 * sd, bias + 1.96 * sd
        normal = predictions.binary_label.to_numpy(int) == 0
        axis.scatter(mean[normal], difference[normal], s=18, alpha=0.64,
                     color="#4C78A8", edgecolors="none", label="Normal side")
        axis.scatter(mean[~normal], difference[~normal], s=18, alpha=0.68,
                     color="#E15759", edgecolors="none", label="Abnormal side")
        axis.axhline(bias, color="#343A40", linewidth=1.3, label="Bias")
        for value in (lower, upper):
            axis.axhline(value, color="#278079", linestyle="--", linewidth=1.2)
        limits = np.concatenate((difference, [lower, upper]))
        span = float(limits.max() - limits.min())
        padding = max(0.06 * span, 0.05)
        axis.set_ylim(float(limits.min() - padding), float(limits.max() + padding))
        axis.set_xlabel(f"Mean of true and predicted ({TASK_UNITS[target]})")
        axis.set_ylabel(f"Predicted minus true ({TASK_UNITS[target]})")
        axis.set_title(
            f"{TASK_LABELS[target]} | n={len(predictions)}\n"
            f"bias={bias:.2f}, limits=[{lower:.2f}, {upper:.2f}]",
        )
        axis.grid(alpha=0.18)
        summary.append({
            "target": target, "n_videos": len(predictions),
            "bias_pred_minus_true": bias, "difference_sd_ddof1": sd,
            "lower_limit_bias_minus_1_96_sd": lower,
            "upper_limit_bias_plus_1_96_sd": upper,
        })
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="#278079", linestyle="--"))
    labels.append("Bias +/- 1.96 SD")
    figure.legend(handles, labels, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, 1.005), frameon=False)
    figure.suptitle(f"{output_dir.name.replace('_', ' ')} | video-level test Bland-Altman", y=1.045,
                    fontsize=14)
    figure.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / "test_bland_altman.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(summary).to_csv(output_dir / "test_bland_altman_summary.csv", index=False)
    print(f"[bland-altman-complete] {figure_dir / 'test_bland_altman.png'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.output_dir)


if __name__ == "__main__":
    main()
