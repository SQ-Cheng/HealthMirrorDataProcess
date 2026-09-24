"""Plot history-only results and the controlled three-pathway comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGETS


TARGET_LABELS = {
    "oxyhemoglobin_fraction": "O2Hb fraction",
    "lactate_high": "Lactate",
    "urea_high": "Urea",
    "total_bilirubin_high": "Total bilirubin",
    "platelet_count_low": "Platelets",
    "hemoglobin_low": "Hemoglobin",
    "aa_po2_ratio_low": "A/a PO2 ratio",
    "creatinine_high": "Creatinine",
}
TARGET_UNITS = {
    "oxyhemoglobin_fraction": "%",
    "lactate_high": "mmol/L",
    "urea_high": "mmol/L",
    "total_bilirubin_high": "umol/L",
    "platelet_count_low": "10^9/L",
    "hemoglobin_low": "g/L",
    "aa_po2_ratio_low": "%",
    "creatinine_high": "umol/L",
}
VARIANTS = (
    ("face_history_efficientnet_b0", "Face+history EfficientNet", "#4C78A8"),
    ("face_only_efficientnet_b0", "Face-only EfficientNet", "#E15759"),
    ("history_only_head32", "History only", "#59A14F"),
)
METRICS = (
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("pearson_r", "Pearson r"),
    ("r2", "R2"),
    ("sign_roc_auc", "Threshold AUC"),
    ("sign_balanced_accuracy", "Threshold bACC"),
)


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


def _one_test_row(frame, architecture, target, source, variant):
    selected = frame.loc[
        frame["architecture"].eq(architecture)
        & frame["target"].eq(target)
        & frame["split"].eq("test")
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Missing comparison metrics for {source}/{architecture}/{target}"
        )
    row = selected.iloc[0].to_dict()
    row["variant"] = variant
    row["source_experiment"] = source
    return row


def _comparison(output_dir, face_history_dir, face_only_dir):
    current = pd.read_csv(output_dir / "metrics_all.csv")
    face_history = pd.read_csv(face_history_dir / "metrics_all.csv")
    face_only = pd.read_csv(face_only_dir / "metrics_all.csv")
    rows = []
    for target in TARGETS:
        for architecture in ("efficientnet_b0",):
            rows.append(_one_test_row(
                face_history, architecture, target, "face_plus_history",
                f"face_history_{architecture}",
            ))
            rows.append(_one_test_row(
                face_only, architecture, target, "face_only",
                f"face_only_{architecture}",
            ))
        rows.append(_one_test_row(
            current, "history_only_head32", target, "history_only",
            "history_only_head32",
        ))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    return comparison


def _plot_training(history, figure_dir):
    figure, axes = plt.subplots(
        len(TARGETS), 2, figsize=(13, 3.2 * len(TARGETS)), squeeze=False
    )
    for row, target in enumerate(TARGETS):
        selected = history.loc[history["target"].eq(target)].sort_values("global_epoch")
        for stage, group in selected.groupby("stage", sort=False):
            axes[row, 0].plot(
                group.global_epoch, group.train_eval_loss, label=f"{stage} train"
            )
            axes[row, 0].plot(
                group.global_epoch, group.val_loss, linestyle="--", label=f"{stage} val"
            )
            axes[row, 1].plot(
                group.global_epoch, group.train_mae, label=f"{stage} train"
            )
            axes[row, 1].plot(
                group.global_epoch, group.val_mae, linestyle="--", label=f"{stage} val"
            )
        axes[row, 0].set_ylabel("SmoothL1 loss")
        axes[row, 1].set_ylabel(f"MAE ({TARGET_UNITS[target]})")
        for column in range(2):
            axes[row, column].set_xlabel("Global epoch")
            axes[row, column].set_title(TARGET_LABELS[target])
            axes[row, column].grid(alpha=0.22)
            axes[row, column].legend(fontsize=7)
    figure.suptitle("History-only Head32 raw-value regression", fontsize=14)
    figure.tight_layout()
    figure.savefig(figure_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_predictions(output_dir, figure_dir):
    rows, columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False
    )
    for column, target in enumerate(TARGETS):
        predictions = pd.read_csv(
            output_dir / "runs" / target / "video_predictions.csv"
        )
        predictions = predictions.loc[predictions["split"].eq("test")]
        axis = axes.flat[column]
        abnormal = predictions["binary_label"].eq(1)
        axis.scatter(
            predictions.loc[~abnormal, "y_true"],
            predictions.loc[~abnormal, "y_pred"],
            s=18, alpha=0.62, color="#4C78A8", label="normal",
        )
        axis.scatter(
            predictions.loc[abnormal, "y_true"],
            predictions.loc[abnormal, "y_pred"],
            s=18, alpha=0.68, color="#E15759", label="abnormal",
        )
        low = min(predictions.y_true.min(), predictions.y_pred.min())
        high = max(predictions.y_true.max(), predictions.y_pred.max())
        axis.plot((low, high), (low, high), color="#333333", linestyle="--")
        axis.set_xlabel(f"True ({TARGET_UNITS[target]})")
        axis.set_ylabel(f"Predicted ({TARGET_UNITS[target]})")
        axis.set_title(f"{TARGET_LABELS[target]} | test n={len(predictions)}")
        axis.grid(alpha=0.2)
        axis.legend()
    for axis in axes.flat[len(TARGETS):]:
        axis.axis("off")
    figure.suptitle("History-only video-level test predictions", fontsize=14)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "test_predicted_vs_true.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def _plot_comparison(comparison, figure_dir):
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    x = np.arange(len(TARGETS))
    width = 0.24
    offsets = np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2
    for axis, (metric, label) in zip(axes.flat, METRICS):
        for offset, (variant, variant_label, color) in zip(offsets, VARIANTS):
            values = []
            for target in TARGETS:
                selected = comparison.loc[
                    comparison["variant"].eq(variant)
                    & comparison["target"].eq(target), metric
                ]
                if len(selected) != 1:
                    raise RuntimeError(f"Incomplete comparison for {variant}/{target}")
                values.append(float(selected.iloc[0]))
            axis.bar(x + offset * width, values, width, color=color, label=variant_label)
        axis.set_xticks(x, [TARGET_LABELS[target] for target in TARGETS])
        axis.set_ylabel(label)
        axis.set_title(f"Test {label}")
        axis.axhline(0, color="#777777", linestyle=":", linewidth=0.7)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=6)
    figure.suptitle(
        "Controlled comparison: face+history, face-only, and history-only",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "three_pathway_model_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(output_dir, face_history_dir, face_only_dir):
    output_dir = Path(output_dir)
    face_history_dir = Path(face_history_dir)
    face_only_dir = Path(face_only_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _style()
    history = pd.read_csv(output_dir / "history_all.csv")
    comparison = _comparison(output_dir, face_history_dir, face_only_dir)
    _plot_training(history, figure_dir)
    _plot_predictions(output_dir, figure_dir)
    _plot_comparison(comparison, figure_dir)
    print(f"Saved 3 result figures to {figure_dir}", flush=True)
