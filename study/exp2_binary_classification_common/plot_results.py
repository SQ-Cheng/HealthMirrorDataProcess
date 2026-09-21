"""Create per-experiment and three-way binary-classification summaries."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import EXPERIMENT_DIRS, MODALITIES, TARGETS


DISPLAY = {
    "oxyhemoglobin_fraction": "O2Hb fraction",
    "lactate_high": "Lactate",
    "urea_high": "Urea",
    "total_bilirubin_high": "Total bilirubin",
    "platelet_count_low": "Platelets",
    "hemoglobin_low": "Hemoglobin",
    "aa_po2_ratio_low": "A/a PO2 ratio",
    "creatinine_high": "Creatinine",
}
MODALITY_DISPLAY = {
    "face_history": "Face + history",
    "face_only": "Face only",
    "history_only": "History only",
}
COLORS = ("#2F6B8A", "#D9822B", "#5B8E3E")


def _run_dir(modality, target):
    root = EXPERIMENT_DIRS[modality] / "outputs/runs"
    return root / "efficientnet_b0" / target if modality != "history_only" else root / target


def collect():
    frames = []
    for modality in MODALITIES:
        for target in TARGETS:
            path = _run_dir(modality, target) / "metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame["modality"] = modality
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_all():
    all_metrics = collect()
    for modality in MODALITIES:
        output_dir = EXPERIMENT_DIRS[modality] / "outputs"
        figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
        metrics = all_metrics[all_metrics.modality.eq(modality)].copy()
        metrics.to_csv(output_dir / "metrics_all.csv", index=False)
        test = metrics[metrics.split.eq("test")].set_index("target").loc[list(TARGETS)]
        figure, axes = plt.subplots(2, 2, figsize=(16, 9))
        for axis, (metric, title) in zip(axes.flat, (
            ("balanced_accuracy", "Balanced accuracy"),
            ("roc_auc", "ROC-AUC"),
            ("f1", "F1"),
            ("average_precision", "Average precision"),
        )):
            values = test[metric].to_numpy(float)
            bars = axis.bar(np.arange(len(TARGETS)), values, color="#2F6B8A")
            axis.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
            axis.set_xticks(np.arange(len(TARGETS)), [DISPLAY[x] for x in TARGETS], rotation=35, ha="right")
            axis.set_ylim(0, 1.05); axis.set_title(title); axis.grid(axis="y", alpha=.25)
        figure.suptitle(f"{MODALITY_DISPLAY[modality]} true binary classification")
        figure.tight_layout(); figure.savefig(
            figure_dir / "test_classification_metrics.png", dpi=180, bbox_inches="tight"
        ); plt.close(figure)
        (output_dir / "experiment_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "task_type": "true_binary_classification",
            "modality": modality,
            "targets": list(TARGETS),
            "target_replacement": "troponin_high replaced by total_bilirubin_high",
            "comparison_contract": (
                "same task records and patient splits as regression for seven targets; "
                "bilirubin built with the same matching and split-selection policy"
            ),
        }, indent=2), encoding="utf-8")

    test = all_metrics[all_metrics.split.eq("test")].copy()
    rows = []
    for target in TARGETS:
        for modality in MODALITIES:
            row = test[test.target.eq(target) & test.modality.eq(modality)].iloc[0]
            rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison_path = EXPERIMENT_DIRS["face_history"] / "outputs/three_way_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    x = np.arange(len(TARGETS)); width = .25
    for axis, metric, title in zip(
        axes, ("balanced_accuracy", "roc_auc"),
        ("Test balanced accuracy", "Test ROC-AUC"),
    ):
        for index, modality in enumerate(MODALITIES):
            selected = comparison[comparison.modality.eq(modality)].set_index("target").loc[list(TARGETS)]
            values = selected[metric].to_numpy(float)
            bars = axis.bar(
                x + (index - 1) * width, values, width,
                color=COLORS[index], label=MODALITY_DISPLAY[modality],
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=6, rotation=90, padding=2)
        axis.set_ylim(0, 1.05); axis.set_title(title); axis.grid(axis="y", alpha=.25)
        axis.legend()
    axes[-1].set_xticks(x, [DISPLAY[t] for t in TARGETS], rotation=30, ha="right")
    figure.suptitle("Exp2 true binary classification: controlled three-pathway comparison")
    figure.tight_layout()
    figure_dir = EXPERIMENT_DIRS["face_history"] / "outputs/figures"
    figure.savefig(figure_dir / "three_way_classification_comparison.png", dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(f"[plots-complete] comparison={comparison_path}", flush=True)


if __name__ == "__main__":
    plot_all()
