"""Compare paired Exp5 against post-only and pre-only controlled ablations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .models import build_ablation_model, build_model, parameter_counts


RUNS = {
    "Paired dual encoder": OUTPUT_DIR,
    "Post only": OUTPUT_DIR / "ablations" / "post_only",
    "Pre only": OUTPUT_DIR / "ablations" / "pre_only",
}
COLORS = ("#2F6B8A", "#D9822B", "#5B8E3E")


def _bar_group(axis, frame, metrics, title):
    x = np.arange(len(metrics)); width = 0.24
    for index, (name, row) in enumerate(frame.iterrows()):
        values = [row[metric] for metric in metrics]
        bars = axis.bar(x + (index - 1) * width, values, width, label=name, color=COLORS[index])
        axis.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    axis.set_xticks(x, metrics); axis.set_title(title); axis.axhline(0, color="#666", lw=.8)
    axis.grid(axis="y", alpha=.2); axis.legend(fontsize=8)


def plot_comparison(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    rows, predictions = [], {}
    for name, run_dir in RUNS.items():
        metrics_path = run_dir / "metrics.csv"
        predictions_path = run_dir / "video_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Incomplete run: {run_dir}")
        metrics = pd.read_csv(metrics_path)
        test = metrics[metrics.split.eq("test")].iloc[0]
        if run_dir == OUTPUT_DIR:
            model, _ = build_model()
        else:
            model, _ = build_ablation_model(run_dir.name)
        parameters = parameter_counts(model)[0]
        del model
        rows.append({
            "model": name, "parameters": parameters,
            "test_videos": int(test["n"]), **test.to_dict(),
        })
        pred = pd.read_csv(
            predictions_path, dtype={"hospital_id": str, "video_id": str}
        )
        pred = pred[pred.split.eq("test")].sort_values(
            ["hospital_id", "video_id"]
        ).reset_index(drop=True)
        predictions[name] = pred
    comparison = pd.DataFrame(rows).set_index("model")
    comparison.to_csv(output_dir / "ablation_comparison.csv")
    pd.concat(
        [pred.assign(model=name) for name, pred in predictions.items()],
        ignore_index=True,
    ).to_csv(output_dir / "ablation_test_predictions.csv", index=False)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    _bar_group(axes[0, 0], comparison, ("mae", "rmse"), "Test error (lower is better)")
    _bar_group(axes[0, 1], comparison, ("r2", "explained_variance"), "Test variance fit")
    _bar_group(axes[0, 2], comparison, ("pearson_r", "spearman_r"), "Test correlation")
    for axis, ((name, pred), color) in zip(axes[1], zip(predictions.items(), COLORS)):
        axis.scatter(pred.y_true, pred.y_pred, s=24, alpha=.72, color=color)
        upper = max(float(pred.y_true.max()), float(pred.y_pred.max())) * 1.05
        axis.plot([0, upper], [0, upper], "--", color="#555", lw=1)
        row = comparison.loc[name]
        parameters_m = row.parameters / 1e6
        axis.set(
            xlim=(0, upper), ylim=(0, upper), xlabel="True trajectory deviation",
            ylabel="Predicted trajectory deviation",
            title=(
                f"{name} ({parameters_m:.2f}M params, n={int(row.test_videos)})\n"
                f"MAE={row.mae:.3f}, R2={row.r2:.3f}, r={row.pearson_r:.3f}"
            ),
        )
        axis.grid(alpha=.2)
    figure.suptitle(
        "Exp5 paired vs single-input protocols\n"
        "Each protocol uses its maximum eligible cohort; test cohorts may differ",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "ablation_paired_vs_single_input.png"
    figure.savefig(path, dpi=190, bbox_inches="tight"); plt.close(figure)
    print(f"[comparison-complete] table={output_dir / 'ablation_comparison.csv'} figure={path}", flush=True)


if __name__ == "__main__":
    plot_comparison()
