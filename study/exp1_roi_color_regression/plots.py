"""ROI/regressor comparisons, original-unit scatter plots and MLP histories."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape, target_grid_figsize
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from .preview_rois import HERE


def panels(targets):
    rows, columns = target_grid_shape(len(targets))
    fig, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False, constrained_layout=True)
    for axis in axes.flat[len(targets):]:
        axis.set_visible(False)
    return fig, axes


def save(figure, path):
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix("." + extension), dpi=180)
    plt.close(figure)


def plot_results(protocol, modes):
    figures = HERE / "outputs/figures/results"
    figures.mkdir(parents=True, exist_ok=True)
    tables = HERE / "outputs/tables"
    metrics = pd.read_csv(tables / "metrics_all.csv")
    test = metrics.loc[metrics.split.eq("test")]
    targets = protocol["targets"]
    for metric, title in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        fig, axes = panels(targets)
        for axis, target in zip(axes.flat, targets):
            part = test.loc[test.target.eq(target)].set_index(["roi_mode", "kind"])
            x = np.arange(len(modes))
            for offset, kind, color in ((-.19, "ridge", "#64737c"), (.19, "mlp", "#2878b5")):
                axis.bar(x + offset, [part.loc[(mode, kind), metric] for mode in modes], width=.38, color=color, label=kind.upper())
            axis.set_xticks(x, [name.replace("image_", "").replace("_", " ") for name in modes], rotation=35, ha="right", fontsize=7)
            axis.set(title=TASK_LABELS[target], ylabel=title + (f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.axhline(0, color="#888888", linewidth=.6)
            axis.grid(axis="y", alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle(f"Local native color regression | same held-out videos | {title}")
        save(fig, figures / f"roi_model_{metric}")
    for mode in modes:
        for kind in ("mlp", "ridge"):
            fig, axes = panels(targets)
            for axis, target in zip(axes.flat, targets):
                run = HERE / f"outputs/models/{kind}/{mode}/{target}"
                frame = pd.read_csv(run / "predictions.csv").query("split == 'test'")
                actual, predicted = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
                axis.scatter(actual, predicted, s=13, alpha=.5, color="#2878b5", edgecolors="none")
                lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
                axis.plot([lo, hi], [lo, hi], "--", color="#64737c")
                if np.ptp(actual):
                    slope, intercept = np.polyfit(actual, predicted, 1)
                    span = np.array([actual.min(), actual.max()])
                    axis.plot(span, slope * span + intercept, color="#cb6547")
                axis.set(title=f"{TASK_LABELS[target]} | n={len(frame)}", xlabel=f"Measured ({TASK_UNITS[target]})", ylabel=f"Predicted ({TASK_UNITS[target]})")
                axis.grid(alpha=.2)
            fig.suptitle(f"{kind.upper()} | {mode} | test prediction")
            save(fig, figures / f"{kind}_{mode}_scatter")
        fig, axes = panels(targets)
        for axis, target in zip(axes.flat, targets):
            history = pd.read_csv(HERE / f"outputs/models/mlp/{mode}/{target}/history.csv")
            axis.plot(history.epoch, history.train_loss, color="#2878b5", label="Train loss")
            axis.plot(history.epoch, history.val_loss, color="#cb6547", label="Val loss")
            twin = axis.twinx()
            twin.plot(history.epoch, history.val_pearson_r, color="#278245", label="Val r")
            finite = history.val_pearson_r.to_numpy(float)
            finite = finite[np.isfinite(finite)]
            if len(finite):
                pad = max(np.ptp(finite) * .15, .04)
                twin.set_ylim(max(-1, finite.min() - pad), min(1, finite.max() + pad))
            axis.set(title=TASK_LABELS[target], xlabel="Epoch", ylabel="SmoothL1")
            twin.set_ylabel("Pearson r")
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
        fig.suptitle(f"MLP native-color training | {mode}")
        save(fig, figures / f"mlp_{mode}_history")
    print(f"[plots-complete] {figures}", flush=True)
