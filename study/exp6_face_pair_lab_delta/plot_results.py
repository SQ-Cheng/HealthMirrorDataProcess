"""Generate experiment-level Exp6 result figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape


DISPLAY = {
    "oxyhemoglobin_fraction": "O2Hb fraction",
    "lactate_high": "Lactate",
    "urea_high": "Urea",
    "troponin_high": "Troponin",
    "platelet_count_low": "Platelet count",
    "hemoglobin_low": "Hemoglobin",
    "aa_po2_ratio_low": "A/a PO2 ratio",
    "creatinine_high": "Creatinine",
    "total_bilirubin_high": "Total bilirubin",
}


def plot_results(output_dir):
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    test = metrics.loc[metrics.split.eq("test")].copy()
    test["label"] = test.target.map(DISPLAY).fillna(test.target)
    order = [DISPLAY[target] for target in DISPLAY if target in set(test.target)]
    test["label"] = pd.Categorical(test.label, categories=order, ordered=True)
    test = test.sort_values("label")

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, metric, title, baseline in (
        (axes[0, 0], "r2", "R2", 0.0),
        (axes[0, 1], "pearson_r", "Pearson r", 0.0),
        (axes[1, 0], "direction_balanced_accuracy", "Direction bACC", 0.5),
        (axes[1, 1], "direction_roc_auc", "Direction ROC AUC", 0.5),
    ):
        values = test[metric].to_numpy(float)
        colors = ["#237a57" if value >= baseline else "#b84a3a" for value in values]
        axis.barh(test.label.astype(str), values, color=colors)
        axis.axvline(baseline, color="#333333", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        for y, value in enumerate(values):
            axis.text(value, y, f" {value:.3f}", va="center", fontsize=8)
    figure.suptitle("Exp6 paired-face laboratory delta prediction: test performance")
    figure.tight_layout()
    figure.savefig(figure_dir / "test_performance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    histories = []
    for path in sorted((output_dir / "runs").glob("*/history.csv")):
        frame = pd.read_csv(path)
        histories.append(frame)
    if histories:
        history = pd.concat(histories, ignore_index=True)
        targets = list(history.target.drop_duplicates())
        rows, columns = target_grid_shape(len(targets))
        figure, axes = plt.subplots(
            rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False
        )
        for axis, target in zip(axes.flat, targets):
            subset = history.loc[history.target.eq(target)]
            axis.plot(subset.global_epoch, subset.train_mae, label="train")
            axis.plot(subset.global_epoch, subset.val_mae, label="validation")
            for boundary in subset.loc[
                subset.stage.ne(subset.stage.shift()), "global_epoch"
            ].iloc[1:]:
                axis.axvline(boundary - 0.5, color="#777777", linestyle=":")
            axis.set_title(DISPLAY.get(target, target))
            axis.set_xlabel("Epoch")
            axis.set_ylabel("MAE (raw unit)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        for axis in axes.flat[len(targets):]:
            axis.axis("off")
        figure.suptitle("Exp6 training histories")
        figure.tight_layout()
        figure.savefig(figure_dir / "training_histories.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    predictions = []
    for path in sorted((output_dir / "runs").glob("*/pair_predictions.csv")):
        frame = pd.read_csv(path)
        predictions.append(frame.loc[frame.split.eq("test")])
    if predictions:
        prediction = pd.concat(predictions, ignore_index=True)
        targets = list(prediction.target.drop_duplicates())
        rows, columns = target_grid_shape(len(targets))
        figure, axes = plt.subplots(
            rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False
        )
        for axis, target in zip(axes.flat, targets):
            subset = prediction.loc[prediction.target.eq(target)]
            axis.scatter(subset.y_true, subset.y_pred, s=14, alpha=0.55, color="#276f9f")
            low = float(min(subset.y_true.min(), subset.y_pred.min()))
            high = float(max(subset.y_true.max(), subset.y_pred.max()))
            axis.plot([low, high], [low, high], "--", color="#444444", linewidth=1)
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.axvline(0, color="#999999", linewidth=0.7)
            axis.set_title(DISPLAY.get(target, target))
            axis.set_xlabel("Observed delta")
            axis.set_ylabel("Predicted delta")
            axis.grid(alpha=0.15)
        for axis in axes.flat[len(targets):]:
            axis.axis("off")
        figure.suptitle("Exp6 test-set observed vs predicted laboratory change")
        figure.tight_layout()
        figure.savefig(figure_dir / "observed_vs_predicted.png", dpi=180, bbox_inches="tight")
        plt.close(figure)
    print(f"[plots-complete] directory={figure_dir}", flush=True)
