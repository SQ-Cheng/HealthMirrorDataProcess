"""Create clinician-facing summaries of the top binary laboratory tasks."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
METRIC_FILES = (
    ROOT / "exp2_face_pretrained_allframes_head32" / "outputs" / "metrics_all.csv",
    ROOT
    / "exp2_face_pretrained_allframes_head32_views3"
    / "outputs"
    / "metrics_all.csv",
)
OUTPUT_DIR = ROOT / "figures"

CLINICAL_LABELS = {
    "hemoglobin_low": "Low hemoglobin\n(anemia)",
    "pco2_low": "Low pCO₂\n(hypocapnia)",
    "po2_low": "Low pO₂\n(hypoxemia)",
    "high_blood_pressure": "High blood pressure\n(hypertension)",
    "lactate_high": "High lactate\n(hyperlactatemia)",
}
LAB_TASKS = frozenset(CLINICAL_LABELS)

COLORS = {
    "train": "#1F4E79",
    "test": "#D97732",
}


def load_runs() -> dict[tuple[str, str, str], dict[str, dict[str, float]]]:
    """Return split metrics keyed by source file, architecture, and task."""
    runs: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for metric_file in METRIC_FILES:
        with metric_file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (metric_file.parent.parent.name, row["architecture"], row["target"])
                runs[key][row["split"]] = {
                    "auc": float(row["roc_auc"]),
                    "bacc": float(row["balanced_accuracy"]),
                }
    return runs


def select_top_tasks(
    runs: dict[tuple[str, str, str], dict[str, dict[str, float]]],
    selection_split: str,
) -> list[tuple[str, dict[str, dict[str, float]], float]]:
    """Select one best run per task, then return the three best tasks."""
    by_task: dict[str, list[tuple[tuple[str, str, str], dict[str, dict[str, float]]]]] = (
        defaultdict(list)
    )
    for key, split_metrics in runs.items():
        if key[2] in LAB_TASKS and {"train", "val", "test"}.issubset(split_metrics):
            by_task[key[2]].append((key, split_metrics))

    selected = []
    for task, candidates in by_task.items():
        # Test performance is only a deterministic tie-breaker when selection
        # bACC is identical (notably for the very small pCO2-positive cohort).
        _, best_metrics = max(
            candidates,
            key=lambda item: (
                item[1][selection_split]["bacc"],
                item[1]["test"]["bacc"],
                item[1]["test"]["auc"],
            ),
        )
        selected.append((task, best_metrics, best_metrics[selection_split]["bacc"]))

    return sorted(selected, key=lambda item: item[2], reverse=True)[:3]


def add_value_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#24313D",
        )


def draw_figure(
    selected,
    selection_split: str,
    output_name: str,
) -> None:
    tasks = [CLINICAL_LABELS[task] for task, _, _ in selected]
    selection_scores = [score for _, _, score in selected]
    x = np.arange(len(tasks))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7), sharey=True)
    fig.patch.set_facecolor("white")

    metrics = (("auc", "ROC area under the curve (AUC)"), ("bacc", "Balanced accuracy (bACC)"))
    for ax, (metric_key, panel_title) in zip(axes, metrics):
        train_values = [split_metrics["train"][metric_key] for _, split_metrics, _ in selected]
        test_values = [split_metrics["test"][metric_key] for _, split_metrics, _ in selected]

        train_bars = ax.bar(
            x - width / 2,
            train_values,
            width,
            color=COLORS["train"],
            label="Training cohort",
            zorder=3,
        )
        test_bars = ax.bar(
            x + width / 2,
            test_values,
            width,
            color=COLORS["test"],
            label="Held-out test cohort",
            zorder=3,
        )

        ax.axhline(
            0.5,
            color="#6B7280",
            linewidth=1.2,
            linestyle=(0, (4, 4)),
            zorder=2,
        )
        ax.set_title(panel_title, fontsize=14, weight="bold", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=11)
        ax.set_ylim(0, 1.10)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color="#D8DEE5", linewidth=0.8, alpha=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB4BE")
        add_value_labels(ax, train_bars)
        add_value_labels(ax, test_bars)

    axes[0].set_ylabel("Performance score", fontsize=12)

    criterion = "validation" if selection_split == "val" else "held-out test"
    fig.suptitle(
        f"Top 3 Laboratory Tasks by {criterion.title()} bACC",
        fontsize=19,
        weight="bold",
        color="#172B3A",
        y=0.98,
    )
    fig.text(
        0.5,
        0.905,
        "Training versus held-out test performance for the best result in each task",
        ha="center",
        fontsize=12,
        color="#4B5D6B",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#6B7280",
            linewidth=1.2,
            linestyle=(0, (4, 4)),
        )
    )
    labels.append("Chance level (0.50)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    score_text = "  •  ".join(
        f"{label.replace(chr(10), ' ')}: {score:.2f}"
        for label, score in zip(tasks, selection_scores)
    )
    fig.text(
        0.5,
        0.055,
        f"Selection {criterion} bACC — {score_text}",
        ha="center",
        fontsize=9.5,
        color="#4B5D6B",
    )
    fig.text(
        0.5,
        0.022,
        "Higher values indicate better discrimination; bACC gives equal weight to positive and negative cases.",
        ha="center",
        fontsize=9.5,
        color="#4B5D6B",
    )

    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.72, wspace=0.14)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    runs = load_runs()
    val_top = select_top_tasks(runs, "val")
    test_top = select_top_tasks(runs, "test")
    draw_figure(val_top, "val", "top3_validation_bacc_clinical.png")
    draw_figure(test_top, "test", "top3_test_bacc_clinical.png")


if __name__ == "__main__":
    main()
