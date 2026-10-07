"""Fold-level and pooled held-out figures for the selected 5-fold experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp2_face_pretrained_head32_regression.plot_results import (
    TASK_LABELS, TASK_UNITS,
)

from .selected_5fold_splits import FOLDS, SPLIT_ROOT, TARGETS


def _run_path(root, target, protocol):
    path = Path(root) / "runs"
    return path / target if protocol == "video_regression" else path / "efficientnet_b0" / target


def _pooled_metrics(frame, classification):
    truth = frame.y_true.to_numpy(float)
    if classification:
        probability = frame.y_probability.to_numpy(float)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(truth, probability >= .5)),
            "roc_auc": float(roc_auc_score(truth, probability)),
        }
    predicted = frame.y_pred.to_numpy(float)
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(mean_squared_error(truth, predicted) ** .5),
        "r2": float(r2_score(truth, predicted)),
        "pearson_r": float(np.corrcoef(truth, predicted)[0, 1]),
    }


def plot_fold_classification(output_dir):
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    test = metrics.loc[metrics.split.eq("test")].set_index("target").loc[list(TARGETS)]
    figure, axes = plt.subplots(2, 2, figsize=(16, 9))
    for axis, metric, label in zip(
        axes.flat,
        ("balanced_accuracy", "roc_auc", "f1", "average_precision"),
        ("Balanced accuracy", "ROC-AUC", "F1", "Average precision"),
    ):
        axis.bar(range(len(TARGETS)), test[metric], color="#2F6B8A")
        axis.set_xticks(range(len(TARGETS)), [TASK_LABELS[t] for t in TARGETS],
                        rotation=30, ha="right")
        axis.set_ylim(0, 1.04)
        axis.set_title(label)
        axis.grid(axis="y", alpha=.24)
    figure.suptitle("Face-only binary classification: held-out fold")
    figure.tight_layout()
    figure.savefig(figures / "test_classification_metrics.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)
    from study.exp2_face_pretrained_head32_classification.plot_confusion_matrices import plot_confusion_matrices
    plot_confusion_matrices(output_dir, TARGETS)


def plot_cv(root, protocol, split_root=None):
    root = Path(root)
    split_root = Path(split_root) if split_root is not None else SPLIT_ROOT
    classification = protocol == "face_classification"
    metrics, predictions, summaries = [], [], []
    for fold in range(FOLDS):
        directory = root / f"fold_{fold}"
        if not (directory / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete 5-fold run: {directory}")
        selected = pd.read_csv(directory / "metrics_all.csv")
        selected = selected.loc[selected.split.eq("test")].copy()
        selected.insert(0, "fold", fold)
        metrics.append(selected)
        for target in TARGETS:
            frame = pd.read_csv(
                _run_path(directory, target, protocol) / "video_predictions.csv",
                dtype={"hospital_id": str, "video_id": str},
            )
            frame = frame.loc[frame.split.eq("test")].copy()
            expected = pd.read_csv(
                split_root / f"{target}_fold{fold}.csv",
                dtype={"hospital_id": str, "video_id": str},
            )
            expected = expected.loc[expected.split.eq("test")].set_index("video_id")
            if (frame.video_id.duplicated().any()
                    or set(frame.video_id) != set(expected.index)):
                raise AssertionError(f"Test-video mismatch: {protocol}/{target}/fold{fold}")
            ordered = frame.set_index("video_id").loc[expected.index]
            truth_column = "binary_label" if classification else "raw_value"
            if (not ordered.hospital_id.eq(expected.hospital_id).all()
                    or not np.allclose(ordered.y_true, expected[truth_column],
                                       rtol=0, atol=1e-7)):
                raise AssertionError(f"Test-label mismatch: {protocol}/{target}/fold{fold}")
            frame.insert(0, "fold", fold)
            if "target" in frame:
                if not frame.target.eq(target).all():
                    raise AssertionError(f"Prediction target mismatch: {target}")
            else:
                frame.insert(1, "target", target)
            predictions.append(frame)
    metric_frame = pd.concat(metrics, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if metric_frame.groupby("target").fold.nunique().ne(FOLDS).any():
        raise AssertionError("Missing test fold metrics")
    prediction_frame.to_csv(root / "oof_predictions.csv", index=False)
    metric_frame.to_csv(root / "cv_test_metrics.csv", index=False)
    fields = (("balanced_accuracy", "roc_auc") if classification
              else ("mae", "rmse", "r2", "pearson_r"))
    for target in TARGETS:
        selected = prediction_frame.loc[prediction_frame.target.eq(target)]
        expected = pd.read_csv(split_root / f"{target}_fold0.csv",
                               dtype={"video_id": str, "hospital_id": str})
        if (len(selected) != len(expected) or selected.video_id.duplicated().any()
                or set(selected.video_id) != set(expected.video_id)):
            raise AssertionError(f"OOF coverage mismatch: {protocol}/{target}")
        pooled = _pooled_metrics(selected, classification)
        target_metrics = metric_frame.loc[metric_frame.target.eq(target)]
        for field in fields:
            summaries.append({
                "target": target, "metric": field,
                "fold_mean": float(target_metrics[field].mean()),
                "fold_std": float(target_metrics[field].std(ddof=1)),
                "pooled_oof": pooled[field], "videos": len(selected),
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "cv_summary.csv", index=False)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, columns = target_grid_shape(len(TARGETS))
    for field in (("balanced_accuracy", "roc_auc") if classification
                  else ("mae", "pearson_r")):
        figure, axes = plt.subplots(rows, columns,
                                    figsize=target_grid_figsize(rows, columns),
                                    squeeze=False)
        for axis, target in zip(axes.flat, TARGETS):
            values = metric_frame.loc[metric_frame.target.eq(target)].sort_values("fold")
            axis.plot(values.fold, values[field], "o-", color="#2878B5", linewidth=1.5)
            mean = float(values[field].mean())
            sd = float(values[field].std(ddof=1))
            axis.axhline(mean, color="#D95F02", linestyle="--", linewidth=1)
            axis.set_title(f"{TASK_LABELS[target]} | {mean:.3f} +/- {sd:.3f}")
            axis.set_xticks(range(FOLDS))
            axis.set_xlabel("Held-out fold")
            axis.set_ylabel(f"MAE ({TASK_UNITS[target]})" if field == "mae"
                            else field.replace("_", " "))
            if field in ("roc_auc", "balanced_accuracy"):
                axis.set_ylim(0, 1.05)
            axis.grid(alpha=.2)
        figure.suptitle(f"{protocol}: five-fold held-out {field}", fontsize=15)
        figure.tight_layout()
        figure.savefig(figures / f"cv_test_{field}.png", dpi=180,
                       bbox_inches="tight")
        plt.close(figure)
    print(f"[cv-plots-complete] protocol={protocol} directory={figures}", flush=True)


def plot_split_distributions(split_root=None):
    split_root = Path(split_root) if split_root is not None else SPLIT_ROOT
    figures = split_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(rows, columns,
                                figsize=target_grid_figsize(rows, columns),
                                squeeze=False)
    for axis, target in zip(axes.flat, TARGETS):
        records = pd.read_csv(split_root / f"{target}_fold0.csv",
                              dtype={"hospital_id": str})
        assignment = pd.read_csv(split_root / "patient_folds.csv",
                                 dtype={"hospital_id": str})
        lookup = dict(assignment.loc[assignment.target.eq(target),
                                 ["hospital_id", "fold"]].itertuples(index=False, name=None))
        fold_values = [records.loc[records.hospital_id.astype(str).map(lookup).eq(fold),
                                   "raw_value"].to_numpy(float)
                       for fold in range(FOLDS)]
        axis.boxplot(fold_values, tick_labels=[str(fold) for fold in range(FOLDS)],
                     showfliers=False)
        axis.set_title(TASK_LABELS[target])
        axis.set_xlabel("Held-out fold")
        axis.set_ylabel(TASK_UNITS[target])
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Patient-disjoint five-fold raw laboratory value distributions",
                    fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "test_fold_raw_distributions.png", dpi=180,
                   bbox_inches="tight")
    plt.close(figure)


def plot_regression_comparison(roots, output_dir):
    labels = ("Face", "Face diverse 30/40", "Video R3D-18")
    colors = ("#73808A", "#D95F02", "#2878B5")
    if len(roots) != len(labels):
        raise ValueError("Expected the three selected regression protocols")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for root, label in zip(roots, labels):
        root = Path(root)
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"Incomplete regression protocol: {root}")
        frame = pd.read_csv(root / "cv_summary.csv")
        frame.insert(0, "protocol", label)
        frames.append(frame)
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(output_dir / "regression_cv_comparison.csv", index=False)
    rows, columns = target_grid_shape(len(TARGETS))
    for metric in ("mae", "pearson_r"):
        figure, axes = plt.subplots(rows, columns,
                                    figsize=target_grid_figsize(rows, columns),
                                    squeeze=False)
        for axis, target in zip(axes.flat, TARGETS):
            selected = comparison.loc[
                comparison.target.eq(target) & comparison.metric.eq(metric)
            ].set_index("protocol").loc[list(labels)]
            axis.bar(range(len(labels)), selected.fold_mean,
                     yerr=selected.fold_std, capsize=3, color=colors, width=.65)
            axis.set_xticks(range(len(labels)), labels, rotation=22, ha="right")
            axis.set_title(TASK_LABELS[target])
            axis.set_ylabel(f"Fold MAE ({TASK_UNITS[target]})" if metric == "mae"
                            else "Fold Pearson r")
            axis.grid(axis="y", alpha=.22)
        figure.suptitle(f"Selected five-fold regression comparison: {metric}",
                        fontsize=15)
        figure.tight_layout()
        figure.savefig(output_dir / f"regression_{metric}.png", dpi=180,
                       bbox_inches="tight")
        plt.close(figure)
    print(f"[comparison-plots-complete] directory={output_dir}", flush=True)
