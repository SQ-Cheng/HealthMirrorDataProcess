"""Video-level test and pooled OOF confusion matrices; no model inference."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from study.common.plot_layout import target_grid_figsize, target_grid_shape


HERE = Path(__file__).resolve().parent


def matrix_counts(frame):
    required = {"video_id", "y_true", "y_probability", "y_pred"}
    if required - set(frame):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(frame))}")
    if frame.empty or frame.video_id.duplicated().any():
        raise ValueError("Test predictions must contain each video exactly once")
    truth = frame.y_true.to_numpy(float)
    probability = frame.y_probability.to_numpy(float)
    saved = frame.y_pred.to_numpy(float)
    if (not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
            or not np.isin(truth, [0, 1]).all()):
        raise ValueError("Invalid binary labels or probabilities")
    predicted = (probability >= 0.5).astype(int)
    if not np.array_equal(saved, predicted):
        raise ValueError("Saved predictions differ from the fixed 0.5 decision threshold")
    return confusion_matrix(truth.astype(int), predicted, labels=[0, 1])


def plot_confusion_matrices(output_dir, targets=None, pooled=False, run_root="runs/efficientnet_b0"):
    from study.exp2_binary_classification_common.plot_results import DISPLAY

    output_dir = Path(output_dir)
    targets = tuple(targets or DISPLAY)
    oof = None
    if pooled:
        oof = pd.read_csv(output_dir / "oof_predictions.csv", dtype={"video_id": str})
        if not oof.split.eq("test").all() or set(oof.target) != set(targets):
            raise ValueError("OOF predictions must be test-only and cover the expected targets")
    metrics_path = output_dir / ("cv_test_metrics.csv" if pooled else "metrics_all.csv")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics.split.eq("test")]
    counts, tables = {}, []
    for target in targets:
        if pooled:
            frame = oof.loc[oof.target.eq(target)]
        else:
            path = output_dir / run_root / target / "video_predictions.csv"
            frame = pd.read_csv(path, dtype={"video_id": str})
            frame = frame.loc[frame.split.eq("test")]
        matrix = matrix_counts(frame)
        expected = metrics.loc[metrics.target.eq(target), ["tn", "fp", "fn", "tp"]]
        if expected.empty or not np.array_equal(matrix.ravel(), expected.sum().to_numpy()):
            raise ValueError(f"Confusion counts disagree with saved test metrics: {target}")
        counts[target] = matrix
        tn, fp, fn, tp = matrix.ravel()
        tables.append({
            "target": target, "n": int(matrix.sum()), "decision_threshold": 0.5,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        })
    stem = "oof_confusion_matrices" if pooled else "test_confusion_matrices"
    pd.DataFrame(tables).to_csv(output_dir / f"{stem}_counts.csv", index=False)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    rows, columns = target_grid_shape(len(targets))
    figure, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns),
        squeeze=False, constrained_layout=True,
    )
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    for axis, target in zip(axes.flat, targets):
        matrix = counts[target]
        totals = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, totals, out=np.zeros((2, 2)), where=totals != 0)
        colored = axis.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                percentage = f"{fractions[i, j]:.1%}" if totals[i, 0] else "N/A"
                axis.text(j, i, f"{cell_names[i, j]}: {matrix[i, j]}\n{percentage}",
                          ha="center", va="center", fontsize=12,
                          color="white" if fractions[i, j] > 0.55 else "#182838")
        axis.set_xticks([0, 1], ["Negative (0)", "Positive (1)"], fontsize=9)
        axis.set_yticks([0, 1], ["Negative (0)", "Positive (1)"], fontsize=9)
        axis.set(xlabel="Predicted label", ylabel="True label",
                 title=f"{DISPLAY.get(target, target)} | n={matrix.sum()}")
    for axis in axes.flat[len(targets):]:
        axis.set_visible(False)
    bar = figure.colorbar(colored, ax=list(axes.flat[:len(targets)]), shrink=0.75, pad=0.025)
    bar.set_label("Proportion within true class")
    cohort = "Pooled out-of-fold test videos" if pooled else "Held-out test videos"
    if output_dir.name.startswith("fold_"):
        cohort += f" | {output_dir.name.replace('_', ' ')}"
    figure.suptitle(f"Face-only classification | {cohort} | threshold = 0.50", fontsize=14)
    for extension in ("png", "pdf"):
        figure.savefig(figures / f"{stem}.{extension}", dpi=180)
    plt.close(figure)
    print(f"[confusion-matrices] {figures / (stem + '.png')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        help="An explicit completed output; default scans this study")
    args = parser.parse_args()
    directories = ([args.output] if args.output else
                   sorted({marker.parent for marker in (HERE / "outputs").rglob("COMPLETE")}))
    for directory in directories:
        if not (directory / "COMPLETE").is_file():
            raise RuntimeError(f"Refusing to plot an incomplete result: {directory}")
        pooled = (directory / "oof_predictions.csv").is_file()
        if pooled or (directory / "metrics_all.csv").is_file():
            plot_confusion_matrices(directory, pooled=pooled)


if __name__ == "__main__":
    main()
