"""Plot history-only results and controlled comparisons with face plus history."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ("hemoglobin_low", "po2_low")
TARGET_LABELS = {"hemoglobin_low": "Hemoglobin", "po2_low": "PO2"}
VARIANTS = (
    ("mobilenet_v3_small", "Face+history MobileNet", "#4C78A8"),
    ("efficientnet_b0", "Face+history EfficientNet", "#F28E2B"),
    ("history_only_head32", "History only", "#59A14F"),
)
METRICS = (
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("pearson_r", "Pearson r"),
    ("r2", "R2"),
    ("sign_roc_auc", "Sign AUC"),
    ("sign_balanced_accuracy", "Sign bACC"),
)


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def _comparison(output_dir, reference_dir):
    current = pd.read_csv(output_dir / "metrics_all.csv")
    baseline = pd.read_csv(reference_dir / "metrics_all.csv")
    rows = []
    for target in TARGETS:
        history_row = current.loc[
            current["target"].eq(target) & current["split"].eq("test")
        ]
        if len(history_row) != 1:
            raise RuntimeError(f"Missing history-only test metrics for {target}")
        rows.append(history_row.iloc[0].to_dict())
        for architecture in ("mobilenet_v3_small", "efficientnet_b0"):
            selected = baseline.loc[
                baseline["architecture"].eq(architecture)
                & baseline["target"].eq(target)
                & baseline["split"].eq("test")
            ]
            if len(selected) != 1:
                raise RuntimeError(f"Missing reference metrics for {architecture}/{target}")
            row = selected.iloc[0].to_dict()
            row["model"] = architecture
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    return comparison


def _plot_training(history, figure_dir):
    figure, axes = plt.subplots(len(TARGETS), 2, figsize=(12, 7), squeeze=False)
    for row, target in enumerate(TARGETS):
        selected = history.loc[history["target"].eq(target)].sort_values("epoch")
        axes[row, 0].plot(selected.epoch, selected.train_eval_loss, label="train")
        axes[row, 0].plot(selected.epoch, selected.val_loss, label="validation")
        axes[row, 0].set_ylabel("SmoothL1 loss")
        axes[row, 1].plot(selected.epoch, selected.train_mae, label="train MAE")
        axes[row, 1].plot(selected.epoch, selected.val_mae, label="validation MAE")
        axes[row, 1].plot(
            selected.epoch, selected.val_pearson_r, label="validation Pearson r"
        )
        for column in range(2):
            axes[row, column].set_xlabel("Epoch")
            axes[row, column].set_title(TARGET_LABELS[target])
            axes[row, column].grid(alpha=0.22)
            axes[row, column].legend()
    figure.suptitle("History-only Head32 regression: training history", fontsize=14)
    figure.tight_layout()
    figure.savefig(figure_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_predictions(output_dir, figure_dir):
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(12, 5.2), squeeze=False)
    for column, target in enumerate(TARGETS):
        predictions = pd.read_csv(output_dir / "runs" / target / "video_predictions.csv")
        predictions = predictions.loc[predictions["split"].eq("test")]
        axis = axes[0, column]
        normal = predictions.y_true.lt(0)
        axis.scatter(
            predictions.loc[normal, "y_true"],
            predictions.loc[normal, "y_pred"],
            s=20,
            alpha=0.65,
            color="#4C78A8",
            label="normal side",
        )
        axis.scatter(
            predictions.loc[~normal, "y_true"],
            predictions.loc[~normal, "y_pred"],
            s=20,
            alpha=0.70,
            color="#E15759",
            label="abnormal side",
        )
        low = min(predictions.y_true.min(), predictions.y_pred.min())
        high = max(predictions.y_true.max(), predictions.y_pred.max())
        axis.plot((low, high), (low, high), color="#333333", linestyle="--")
        axis.axhline(0, color="#888888", linewidth=0.7)
        axis.axvline(0, color="#888888", linewidth=0.7)
        axis.set_xlabel("True abnormal score")
        axis.set_ylabel("Predicted abnormal score")
        axis.set_title(f"{TARGET_LABELS[target]} | test n={len(predictions)}")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("History-only video-level test predictions", fontsize=14)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "test_predicted_vs_true.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def _plot_comparison(comparison, figure_dir):
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    x = np.arange(len(TARGETS))
    width = 0.24
    for axis, (metric, label) in zip(axes.flat, METRICS):
        for index, (model, variant_label, color) in enumerate(VARIANTS):
            values = []
            for target in TARGETS:
                selected = comparison.loc[
                    comparison["model"].eq(model) & comparison["target"].eq(target),
                    metric,
                ]
                if len(selected) != 1:
                    raise RuntimeError(f"Incomplete comparison for {model}/{target}")
                values.append(float(selected.iloc[0]))
            bars = axis.bar(
                x + (index - 1) * width,
                values,
                width,
                color=color,
                label=variant_label,
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        axis.set_xticks(x, [TARGET_LABELS[target] for target in TARGETS])
        axis.set_ylabel(label)
        axis.set_title(f"Test {label}")
        axis.axhline(0, color="#777777", linestyle=":", linewidth=0.7)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=7)
    figure.suptitle(
        "Controlled ablation: face + history versus history only", fontsize=15
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "face_history_vs_history_only.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(output_dir, reference_dir):
    output_dir, reference_dir = Path(output_dir), Path(reference_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _style()
    history = pd.read_csv(output_dir / "history_all.csv")
    comparison = _comparison(output_dir, reference_dir)
    _plot_training(history, figure_dir)
    _plot_predictions(output_dir, figure_dir)
    _plot_comparison(comparison, figure_dir)
    print(f"Saved 3 result figures to {figure_dir}", flush=True)
