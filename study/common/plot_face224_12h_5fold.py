"""OOF figures and controlled comparison for the native-224 12h protocols."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import roc_curve

from .plot_layout import target_grid_shape, target_grid_figsize
from .selected_5fold_splits import TARGETS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS


def plot_oof(root, protocol):
    root = Path(root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "oof_predictions.csv")
    rows, columns = target_grid_shape(len(TARGETS))
    fig, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False)
    for ax, target in zip(axes.flat, TARGETS):
        selected = frame.loc[frame.target.eq(target)]
        if protocol == "regression_diverse":
            actual, predicted = selected.y_true.to_numpy(float), selected.y_pred.to_numpy(float)
            ax.scatter(actual, predicted, s=12, alpha=.4, color="#2878B5", edgecolors="none")
            lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1, label="Identity")
            if np.std(actual) > 0:
                fit = linregress(actual, predicted)
                x = np.array([actual.min(), actual.max()])
                ax.plot(x, fit.intercept + fit.slope * x, color="#CB6547", label="Linear fit")
            ax.set(xlabel=f"Measured ({TASK_UNITS[target]})", ylabel=f"Predicted ({TASK_UNITS[target]})")
            ax.legend(fontsize=7)
        else:
            fpr, tpr, _ = roc_curve(selected.y_true, selected.y_probability)
            ax.plot(fpr, tpr, color="#2878B5", lw=1.5)
            ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
            ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f"{TASK_LABELS[target]} | OOF n={len(selected)}")
        ax.grid(alpha=.2)
    fig.tight_layout()
    stem = "oof_predicted_vs_true" if protocol == "regression_diverse" else "oof_roc_curves"
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"{stem}.{extension}", dpi=180)
    plt.close(fig)
    histories = pd.concat([pd.read_csv(root / f"fold_{fold}/history_all.csv").assign(fold=fold) for fold in range(5)], ignore_index=True)
    histories.to_csv(root / "history_all.csv", index=False)
    if protocol != "regression_diverse":
        for fold in range(5):
            fig, axes = plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False)
            for ax, target in zip(axes.flat, TARGETS):
                selected = histories.loc[histories.fold.eq(fold) & histories.target.eq(target)]
                ax.plot(selected.global_epoch, selected.train_loss, label="Train", color="#2878B5")
                ax.plot(selected.global_epoch, selected.val_loss, label="Validation", color="#CB6547")
                ax.set(title=TASK_LABELS[target], xlabel="Epoch", ylabel="Weighted BCE loss")
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(root / f"fold_{fold}/figures/training_history.png", dpi=180)
            plt.close(fig)


def plot_classification_comparison(outputs, figures):
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    protocols = ("classification_standard", "classification_diverse")
    labels = ("Standard", "Patient-diverse 30/40")
    frames = []
    for protocol in protocols:
        root = outputs[protocol]
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete classification protocol: {root}")
        frames.append(pd.read_csv(root / "cv_summary.csv").assign(protocol=protocol))
    first = pd.read_csv(outputs[protocols[0]] / "oof_predictions.csv", dtype={"hospital_id": str, "video_id": str})
    second = pd.read_csv(outputs[protocols[1]] / "oof_predictions.csv", dtype={"hospital_id": str, "video_id": str})
    identity = ["target", "fold", "video_id", "hospital_id"]
    pd.testing.assert_frame_equal(first.sort_values(identity)[identity + ["y_true"]].reset_index(drop=True),
                                  second.sort_values(identity)[identity + ["y_true"]].reset_index(drop=True), check_dtype=False)
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(figures.parent / "classification_cv_comparison.csv", index=False)
    for metric, ylabel in (("balanced_accuracy", "Balanced accuracy"), ("roc_auc", "AUROC")):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
        for ax, mode in zip(axes, ("fold_mean", "pooled_oof")):
            x = np.arange(len(TARGETS))
            for i, (protocol, label, color) in enumerate(zip(protocols, labels, ("#2878B5", "#CB6547"))):
                selected = comparison.loc[comparison.protocol.eq(protocol) & comparison.metric.eq(metric)].set_index("target").loc[list(TARGETS)]
                ax.bar(x + (i - .5) * .38, selected[mode], width=.36,
                       yerr=selected.fold_std if mode == "fold_mean" else None,
                       capsize=3, label=label, color=color)
            ax.set(title="Fold mean +/- SD" if mode == "fold_mean" else "Pooled out-of-fold", ylabel=ylabel, ylim=(0, 1.05))
            ax.set_xticks(x, [TASK_LABELS[t] for t in TARGETS], rotation=35, ha="right")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
        fig.suptitle(f"Native 224, 12h: same patient folds | {ylabel}")
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"classification_{metric}_comparison.{extension}", dpi=180)
        plt.close(fig)
