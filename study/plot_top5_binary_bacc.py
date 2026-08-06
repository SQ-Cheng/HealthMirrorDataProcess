"""Plot train/validation metrics for the top binary-classification runs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
EXPERIMENTS = {
    "Views5": ROOT
    / "exp2_face_pretrained_allframes_head32"
    / "outputs"
    / "metrics_all.csv",
    "Views3": ROOT
    / "exp2_face_pretrained_allframes_head32_views3"
    / "outputs"
    / "metrics_all.csv",
}
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3",
    "efficientnet_b0": "EfficientNet-B0",
}
TASK_LABELS = {
    "hemoglobin_low": "Hemoglobin low",
    "pco2_low": "pCO₂ low",
    "po2_low": "pO₂ low",
    "high_blood_pressure": "High blood pressure",
    "lactate_high": "Lactate high",
}
SPLIT_COLORS = {"train": "#3B6FB6", "val": "#E58B2A"}


def load_metrics() -> pd.DataFrame:
    frames = []
    for experiment, path in EXPERIMENTS.items():
        frame = pd.read_csv(path)
        frame.insert(0, "experiment", experiment)
        frames.append(frame)

    metrics = pd.concat(frames, ignore_index=True)
    selected = metrics.loc[metrics["split"].isin(["train", "val"])].copy()
    key = ["experiment", "architecture", "target"]
    wide = selected.pivot(
        index=key,
        columns="split",
        values=["balanced_accuracy", "roc_auc"],
    )
    wide.columns = [f"{split}_{metric}" for metric, split in wide.columns]
    wide = wide.reset_index()
    expected = {
        "train_balanced_accuracy",
        "val_balanced_accuracy",
        "train_roc_auc",
        "val_roc_auc",
    }
    if len(wide) != 20 or not expected.issubset(wide.columns):
        raise RuntimeError(
            "Expected 20 complete Views3/Views5 binary runs with train/val metrics."
        )
    if wide[list(expected)].isna().any().any():
        raise RuntimeError("Missing train/validation AUC or balanced accuracy.")
    return wide


def select_top_five(metrics: pd.DataFrame, selection_split: str) -> pd.DataFrame:
    other_split = "train" if selection_split == "val" else "val"
    sort_columns = [
        f"{selection_split}_balanced_accuracy",
        f"{other_split}_balanced_accuracy",
        f"{selection_split}_roc_auc",
        f"{other_split}_roc_auc",
        "experiment",
        "architecture",
        "target",
    ]
    ascending = [False, False, False, False, True, True, True]
    return (
        metrics.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        .head(5)
        .reset_index(drop=True)
    )


def run_label(row: pd.Series, rank: int) -> str:
    return (
        f"{rank}. {row['experiment']} · "
        f"{ARCHITECTURE_LABELS[row['architecture']]} · "
        f"{TASK_LABELS[row['target']]}"
    )


def add_value_labels(axis: plt.Axes, bars) -> None:
    for bar in bars:
        value = bar.get_width()
        axis.text(
            min(value + 0.012, 1.025),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#303030",
        )


def plot_top_five(metrics: pd.DataFrame, selection_split: str) -> Path:
    top = select_top_five(metrics, selection_split)
    labels = [run_label(row, rank) for rank, (_, row) in enumerate(top.iterrows(), 1)]
    positions = np.arange(len(top), dtype=np.float64)
    height = 0.34

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    panels = (
        ("roc_auc", "ROC–AUC"),
        ("balanced_accuracy", "Balanced accuracy"),
    )
    for axis, (metric, title) in zip(axes, panels):
        train_bars = axis.barh(
            positions - height / 2,
            top[f"train_{metric}"],
            height=height,
            color=SPLIT_COLORS["train"],
            alpha=0.92,
            label="Train",
        )
        val_bars = axis.barh(
            positions + height / 2,
            top[f"val_{metric}"],
            height=height,
            color=SPLIT_COLORS["val"],
            alpha=0.92,
            label="Validation",
        )
        axis.axvline(0.5, color="#666666", linestyle="--", linewidth=1, alpha=0.8)
        axis.set_xlim(0, 1.08)
        axis.set_xlabel(title)
        axis.set_title(f"Train vs validation {title}")
        axis.grid(axis="x", alpha=0.22)
        axis.set_axisbelow(True)
        add_value_labels(axis, train_bars)
        add_value_labels(axis, val_bars)

    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    figure.legend(
        [train_bars, val_bars],
        ["Train", "Validation"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
    )
    selection_label = "validation" if selection_split == "val" else "train"
    figure.suptitle(
        f"Top-5 binary runs selected by {selection_label} balanced accuracy",
        fontsize=16,
        fontweight="semibold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.018,
        (
            "Views3 and Views5 final selected checkpoints · "
            "ties resolved by the other split's bACC, then ROC–AUC"
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.115, 1, 0.94))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / f"top5_{selection_split}_bacc_train_val_auc_bacc.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    metrics = load_metrics()
    for split in ("val", "train"):
        output = plot_top_five(metrics, split)
        print(output)


if __name__ == "__main__":
    main()
