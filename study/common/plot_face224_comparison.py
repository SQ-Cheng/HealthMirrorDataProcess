"""Compare native 224 crops with saved legacy results, including paired tests."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)

from .plot_layout import target_grid_shape


LABELS = {"oxyhemoglobin_fraction": "O2Hb fraction", "lactate_high": "Lactate", "urea_high": "Urea",
          "troponin_high": "Troponin", "total_bilirubin_high": "Total bilirubin", "platelet_count_low": "Platelets",
          "hemoglobin_low": "Hemoglobin", "aa_po2_ratio_low": "A/a PO2 ratio", "creatinine_high": "Creatinine"}


def predictions(output, family, target):
    if family == "delta":
        path = output / f"runs/{target}/pair_predictions.csv"
        identity, actual, predicted = ["hospital_id", "pair_id", "first_video_id", "second_video_id"], "raw_delta", "y_pred"
    elif family == "spectral":
        path = output / f"runs/{target}/predictions.csv"
        identity, actual, predicted = ["hospital_id", "video_id"], "raw_value", "predicted_raw_value"
    else:
        path = output / f"runs/efficientnet_b0/{target}/video_predictions.csv"
        identity = ["hospital_id", "video_id"]
        actual, predicted = ("y_true", "y_probability") if family == "classification" else ("raw_value", "y_pred")
    frame = pd.read_csv(path, dtype={key: str for key in identity})
    frame = frame.loc[frame.split.eq("test"), [*identity, actual, predicted]].copy()
    if frame.duplicated(identity).any():
        raise ValueError(f"Duplicate test identity: {path}")
    return frame.rename(columns={actual: "actual", predicted: "predicted"}), identity


def metrics(actual, predicted, binary):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if not len(actual):
        return {"n": 0}
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite comparison predictions")
    if binary:
        labels = predicted >= .5
        two_classes = len(np.unique(actual)) == 2
        return {"n": len(actual), "balanced_accuracy": balanced_accuracy_score(actual, labels),
                "roc_auc": roc_auc_score(actual, predicted) if two_classes else np.nan,
                "f1": f1_score(actual, labels, zero_division=0),
                "average_precision": average_precision_score(actual, predicted) if two_classes else np.nan}
    return {"n": len(actual), "mae": mean_absolute_error(actual, predicted),
            "rmse": np.sqrt(mean_squared_error(actual, predicted)),
            "r2": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
            "pearson_r": float(pearsonr(actual, predicted).statistic)
            if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0 else np.nan}


def plot_comparison(job):
    baseline, output, family = Path(job["baseline"]), Path(job["output"]), job["family"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    baseline_metrics = pd.read_csv(baseline / "metrics_all.csv")
    targets = [target for target in LABELS if target in set(baseline_metrics.target)]
    rows, paired = [], []
    for target in targets:
        old, identity = predictions(baseline, family, target)
        new, _ = predictions(output, family, target)
        common = old.merge(new, on=identity, how="inner", suffixes=("_legacy", "_face224"), validate="one_to_one")
        if not np.allclose(common.actual_legacy, common.actual_face224, rtol=1e-10, atol=1e-8):
            raise ValueError(f"Common held-out labels changed: {target}")
        common["target"] = target
        paired.append(common)
        for cohort, old_frame, new_frame in (
            ("full_test", old, new),
            ("common_test", common.rename(columns={"actual_legacy": "actual", "predicted_legacy": "predicted"}),
             common.rename(columns={"actual_face224": "actual", "predicted_face224": "predicted"})),
        ):
            for source, frame in (("legacy128", old_frame), ("face224", new_frame)):
                rows.append({"target": target, "cohort": cohort, "source": source,
                             **metrics(frame.actual, frame.predicted, family == "classification")})
    result = pd.DataFrame(rows)
    result.to_csv(output / "face224_comparison.csv", index=False)
    pd.concat(paired, ignore_index=True).to_csv(output / "face224_common_test_predictions.csv", index=False)
    names = [LABELS[target] for target in targets]
    for cohort in ("full_test", "common_test"):
        selected = result.loc[result.cohort.eq(cohort)]
        if family == "classification":
            panels = [("balanced_accuracy", "Balanced accuracy"), ("roc_auc", "AUROC"),
                      ("f1", "F1"), ("average_precision", "Average precision")]
            fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        else:
            panels = [("r2", "R2"), ("pearson_r", "Pearson r"), ("mae_ratio", "MAE ratio: native 224 / legacy 128")]
            fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
        x = np.arange(len(targets))
        for ax, (metric, title) in zip(axes.flat, panels):
            if metric == "mae_ratio":
                old = selected.loc[selected.source.eq("legacy128")].set_index("target").reindex(targets)
                new = selected.loc[selected.source.eq("face224")].set_index("target").reindex(targets)
                ax.bar(x, new.mae / old.mae, color="#b96c43")
                ax.axhline(1, color="#777", ls="--", lw=1)
            else:
                for source, shift, color, label in (("legacy128", -.19, "#357b98", "Legacy 128 crops"),
                                                     ("face224", .19, "#b96c43", "Native 224 Kalman crops")):
                    values = selected.loc[selected.source.eq(source)].set_index("target").reindex(targets)
                    ax.bar(x + shift, values[metric] if metric in values else np.full(len(x), np.nan), .36, color=color, label=label)
                if family == "classification":
                    ax.set_ylim(0, 1.05)
            ax.set(title=title, xticks=x, xticklabels=names)
            ax.tick_params(axis="x", labelrotation=35)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes.flat[0].legend(fontsize=8)
        description = "common held-out identities" if cohort == "common_test" else "full held-out cohorts (sizes may differ)"
        fig.suptitle(f"{job['key']}: {description}")
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"face224_vs_legacy_{cohort}.{extension}", dpi=180)
        plt.close(fig)
    if family == "classification":
        test_metrics = pd.read_csv(output / "metrics_all.csv")
        test_metrics = test_metrics.loc[test_metrics.split.eq("test")].set_index("target").reindex(targets)
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
        for ax, (metric, title) in zip(axes.flat, (
            ("balanced_accuracy", "Balanced accuracy"), ("roc_auc", "AUROC"),
            ("f1", "F1"), ("average_precision", "Average precision"),
        )):
            bars = ax.bar(np.arange(len(targets)), test_metrics[metric], color="#357b98")
            ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
            ax.set(title=title, xticks=np.arange(len(targets)), xticklabels=names, ylim=(0, 1.05))
            ax.tick_params(axis="x", labelrotation=35)
            for label in ax.get_xticklabels():
                label.set_ha("right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.suptitle("Face-only native 224 true binary classification")
        fig.savefig(figures / "test_classification_metrics.png", dpi=180)
        plt.close(fig)
        history = pd.read_csv(output / "history_all.csv")
        rows_count, columns = target_grid_shape(len(targets))
        fig, axes = plt.subplots(rows_count, columns, figsize=(5.3 * columns, 3.5 * rows_count), squeeze=False)
        for ax, target in zip(axes.flat, targets):
            frame = history.loc[history.target.eq(target)]
            ax.plot(frame.global_epoch, frame.train_loss, label="Train loss", color="#357b98")
            ax.plot(frame.global_epoch, frame.val_loss, label="Validation loss", color="#b96c43")
            ax.set(title=LABELS[target], xlabel="Epoch", ylabel="Weighted BCE loss")
            ax.legend(fontsize=8)
        for ax in axes.flat[len(targets):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures / "training_history.png", dpi=180)
        plt.close(fig)
    print(f"[comparison-generated] {job['key']} {figures}", flush=True)
