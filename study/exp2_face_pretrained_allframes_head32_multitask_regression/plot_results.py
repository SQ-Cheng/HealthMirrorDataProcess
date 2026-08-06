"""Generate multi-output regression figures and single-task comparisons."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
SINGLE_OUTPUT_DIR = (
    EXP_DIR.parent / "exp2_face_pretrained_head32_regression" / "outputs"
)
FIGURE_DIR = OUTPUT_DIR / "figures"

ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
TASKS = (
    "hemoglobin_low",
    "pco2_low",
    "po2_low",
    "high_blood_pressure",
    "lactate_high",
)
TASK_LABELS = {
    "hemoglobin_low": "Hemoglobin low",
    "pco2_low": "pCO2 low",
    "po2_low": "pO2 low",
    "high_blood_pressure": "High blood pressure",
    "lactate_high": "Lactate high",
}
ARCHITECTURE_COLORS = {
    "mobilenet_v3_small": "#2878B5",
    "efficientnet_b0": "#D95F02",
}
TASK_COLORS = ("#4C78A8", "#F2A541", "#59A14F", "#E15759", "#B279A2")
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F2A541", "test": "#59A14F"}
METHOD_COLORS = {"multi_output": "#2A6F97", "single_task": "#D97706"}
METHOD_LABELS = {
    "multi_output": "Multi-output head",
    "single_task": "Separate single-task models",
}


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _validate_complete(output_dir):
    run_index = pd.read_csv(output_dir / "run_index.csv")
    failed = run_index.loc[~run_index["status"].eq("ok")]
    if not failed.empty:
        raise RuntimeError(f"Incomplete runs in {output_dir}: {failed.to_dict('records')}")


def _stage_boundary(axis, selected):
    finetune = selected.loc[selected["stage"].eq("finetune"), "global_epoch"]
    if not finetune.empty:
        axis.axvline(
            finetune.min() - 0.5,
            color="#666666",
            linestyle=":",
            linewidth=1,
            label="Backbone unfrozen",
        )


def plot_training_curves(history):
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    for row, architecture in enumerate(ARCHITECTURES):
        selected = history.loc[
            history["architecture"].eq(architecture)
        ].sort_values("global_epoch")
        epochs = selected["global_epoch"].to_numpy()

        axis = axes[row, 0]
        axis.plot(epochs, selected["train_eval_loss"], label="Train evaluation")
        axis.plot(epochs, selected["val_loss"], label="Validation")
        _stage_boundary(axis, selected)
        axis.set_ylabel("Task-balanced masked SmoothL1")
        axis.set_title(f"{ARCHITECTURE_LABELS[architecture]} | loss")
        axis.legend()

        axis = axes[row, 1]
        for task, color in zip(TASKS, TASK_COLORS):
            axis.plot(
                epochs,
                selected[f"val_{task}_mae"],
                color=color,
                linewidth=1.2,
                label=TASK_LABELS[task],
            )
        axis.plot(
            epochs,
            selected["val_macro_mae"],
            color="#222222",
            linewidth=2,
            linestyle="--",
            label="Macro",
        )
        _stage_boundary(axis, selected)
        axis.set_ylabel("Video-level validation MAE")
        axis.set_title(f"{ARCHITECTURE_LABELS[architecture]} | task MAE")
        axis.legend(ncol=2)

        axis = axes[row, 2]
        for task, color in zip(TASKS, TASK_COLORS):
            axis.plot(
                epochs,
                selected[f"val_{task}_pearson_r"],
                color=color,
                linewidth=1.2,
                label=TASK_LABELS[task],
            )
        axis.plot(
            epochs,
            selected["val_macro_pearson_r"],
            color="#222222",
            linewidth=2,
            linestyle="--",
            label="Macro",
        )
        _stage_boundary(axis, selected)
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_ylim(-1.05, 1.05)
        axis.set_ylabel("Video-level validation Pearson r")
        axis.set_title(f"{ARCHITECTURE_LABELS[architecture]} | task correlation")
        axis.legend(ncol=2)

        for axis in axes[row]:
            axis.set_xlabel("Global epoch")
            axis.grid(axis="y", alpha=0.23)

    figure.suptitle("Multi-output abnormal-score regression: training history", fontsize=15)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_test_metrics(metrics):
    test = metrics.loc[metrics["split"].eq("test") & metrics["target"].isin(TASKS)]
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    specifications = (
        (axes[0, 0], ("mae", "rmse"), ("MAE", "RMSE"), "Prediction error", None),
        (
            axes[0, 1],
            ("pearson_r", "spearman_r"),
            ("Pearson r", "Spearman r"),
            "Correlation",
            (-1.05, 1.05),
        ),
        (axes[1, 0], ("r2",), ("R2",), "Coefficient of determination", None),
        (
            axes[1, 1],
            ("sign_roc_auc", "sign_balanced_accuracy"),
            ("Sign ROC-AUC", "Sign bACC"),
            "Zero-boundary classification",
            (0, 1.04),
        ),
    )
    for axis, columns, labels, ylabel, ylim in specifications:
        subwidth = width / len(columns)
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            selected = (
                test.loc[test["architecture"].eq(architecture)]
                .set_index("target")
                .loc[list(TASKS)]
            )
            center = x + (architecture_index - 0.5) * width
            for metric_index, (column, label) in enumerate(zip(columns, labels)):
                position = center + (
                    metric_index - (len(columns) - 1) / 2
                ) * subwidth
                axis.bar(
                    position,
                    selected[column],
                    width=subwidth,
                    color=ARCHITECTURE_COLORS[architecture],
                    alpha=0.92 if metric_index == 0 else 0.52,
                    hatch=None if metric_index == 0 else "//",
                    label=f"{ARCHITECTURE_LABELS[architecture]} | {label}",
                )
        axis.axhline(
            0.5 if "sign_roc_auc" in columns else 0,
            color="#555555",
            linestyle="--" if "sign_roc_auc" in columns else ":",
            linewidth=0.9,
        )
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            x, [TASK_LABELS[target] for target in TASKS], rotation=22, ha="right"
        )
        axis.grid(axis="y", alpha=0.24)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Multi-output regression: video-level test performance", fontsize=15)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_split_generalization(metrics):
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.24
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), squeeze=False)
    for row, architecture in enumerate(ARCHITECTURES):
        selected = metrics.loc[
            metrics["architecture"].eq(architecture) & metrics["target"].isin(TASKS)
        ]
        for column, (metric, ylabel) in enumerate(
            (("mae", "Video-level MAE"), ("pearson_r", "Video-level Pearson r"))
        ):
            axis = axes[row, column]
            for offset, split in enumerate(("train", "val", "test")):
                values = (
                    selected.loc[selected["split"].eq(split)]
                    .set_index("target")
                    .loc[list(TASKS), metric]
                )
                axis.bar(
                    x + (offset - 1) * width,
                    values,
                    width=width,
                    color=SPLIT_COLORS[split],
                    alpha=0.88,
                    label={"train": "Train", "val": "Validation", "test": "Test"}[
                        split
                    ],
                )
            axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
            axis.set_xticks(
                x,
                [TASK_LABELS[target] for target in TASKS],
                rotation=24,
                ha="right",
            )
            axis.set_ylabel(ylabel)
            axis.set_title(ARCHITECTURE_LABELS[architecture])
            axis.grid(axis="y", alpha=0.24)
            axis.legend()
    figure.suptitle("Multi-output regression: split generalization", fontsize=15)
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "split_generalization.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_predictions(metrics):
    test_metrics = metrics.loc[
        metrics["split"].eq("test") & metrics["target"].isin(TASKS)
    ].set_index(["architecture", "target"])
    figure, axes = plt.subplots(5, 2, figsize=(13, 22), squeeze=False)
    for row, task in enumerate(TASKS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            predictions = pd.read_csv(
                OUTPUT_DIR / "runs" / architecture / "video_predictions.csv"
            )
            predictions = predictions.loc[
                predictions["split"].eq("test") & predictions["target"].eq(task)
            ]
            y_true = predictions["y_true"].to_numpy(dtype=np.float64)
            y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
            normal = y_true < 0
            axis.scatter(
                y_true[normal],
                y_pred[normal],
                s=22,
                alpha=0.67,
                color="#4C78A8",
                edgecolors="none",
                label="Normal side",
            )
            axis.scatter(
                y_true[~normal],
                y_pred[~normal],
                s=22,
                alpha=0.72,
                color="#E15759",
                edgecolors="none",
                label="Abnormal/boundary side",
            )
            lower = float(min(y_true.min(), y_pred.min()))
            upper = float(max(y_true.max(), y_pred.max()))
            padding = max((upper - lower) * 0.06, 0.05)
            limits = (lower - padding, upper + padding)
            axis.plot(limits, limits, color="#333333", linestyle="--", linewidth=1)
            axis.axhline(0, color="#888888", linewidth=0.7)
            axis.axvline(0, color="#888888", linewidth=0.7)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("True abnormal score")
            axis.set_ylabel("Predicted abnormal score")
            row_metrics = test_metrics.loc[(architecture, task)]
            axis.set_title(
                f"{TASK_LABELS[task]} | {ARCHITECTURE_LABELS[architecture]}\n"
                f"n={int(row_metrics['n'])}, MAE={row_metrics['mae']:.3f}, "
                f"r={row_metrics['pearson_r']:.3f}"
            )
            axis.grid(alpha=0.18)
            axis.legend(loc="best")
    figure.suptitle(
        "Multi-output regression: video-level test predictions", fontsize=15, y=1.002
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "test_predicted_vs_true.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def _reported_comparison(multi_metrics, single_metrics):
    frames = []
    for method, frame in (
        ("multi_output", multi_metrics),
        ("single_task", single_metrics),
    ):
        selected = frame.loc[
            frame["split"].eq("test") & frame["target"].isin(TASKS)
        ].copy()
        selected.insert(0, "method", method)
        frames.append(selected)
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(FIGURE_DIR / "comparison_reported_test_metrics.csv", index=False)
    return comparison


def plot_reported_comparison(comparison):
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    specifications = (
        ("mae", "Video-level MAE", None),
        ("pearson_r", "Video-level Pearson r", (-1.05, 1.05)),
        ("sign_roc_auc", "Sign ROC-AUC", (0, 1.04)),
    )
    for row, architecture in enumerate(ARCHITECTURES):
        for column, (metric, ylabel, ylim) in enumerate(specifications):
            axis = axes[row, column]
            for method_index, method in enumerate(("multi_output", "single_task")):
                selected = (
                    comparison.loc[
                        comparison["architecture"].eq(architecture)
                        & comparison["method"].eq(method)
                    ]
                    .set_index("target")
                    .loc[list(TASKS)]
                )
                axis.bar(
                    x + (method_index - 0.5) * width,
                    selected[metric],
                    width=width,
                    color=METHOD_COLORS[method],
                    alpha=0.9,
                    label=METHOD_LABELS[method],
                )
            axis.axhline(
                0.5 if metric == "sign_roc_auc" else 0,
                color="#666666",
                linestyle="--" if metric == "sign_roc_auc" else ":",
                linewidth=0.8,
            )
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.set_xticks(
                x,
                [TASK_LABELS[target] for target in TASKS],
                rotation=24,
                ha="right",
            )
            axis.set_ylabel(ylabel)
            axis.set_title(ARCHITECTURE_LABELS[architecture])
            axis.grid(axis="y", alpha=0.24)
            axis.legend()
    figure.suptitle(
        "Reported test performance: multi-output vs single-task\n"
        "Caution: each experiment used its own patient-level test assignment",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "comparison_reported_test_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _common_test_comparison():
    rows = []
    for architecture in ARCHITECTURES:
        multi_predictions = pd.read_csv(
            OUTPUT_DIR / "runs" / architecture / "video_predictions.csv"
        )
        for task in TASKS:
            multi = multi_predictions.loc[
                multi_predictions["split"].eq("test")
                & multi_predictions["target"].eq(task),
                ["video_id", "y_true", "y_pred"],
            ]
            single = pd.read_csv(
                SINGLE_OUTPUT_DIR
                / "runs"
                / architecture
                / task
                / "video_predictions.csv"
            )
            single = single.loc[
                single["split"].eq("test"), ["video_id", "y_true", "y_pred"]
            ]
            common = multi.merge(
                single,
                on="video_id",
                how="inner",
                suffixes=("_multi", "_single"),
                validate="one_to_one",
            )
            if not np.allclose(common["y_true_multi"], common["y_true_single"]):
                raise ValueError(f"Target mismatch for {architecture}/{task}")
            y_true = common["y_true_multi"].to_numpy(dtype=np.float64)
            for method, prediction_column in (
                ("multi_output", "y_pred_multi"),
                ("single_task", "y_pred_single"),
            ):
                y_pred = common[prediction_column].to_numpy(dtype=np.float64)
                pearson = (
                    float(np.corrcoef(y_true, y_pred)[0, 1])
                    if len(y_true) > 1
                    and np.std(y_true) > 0
                    and np.std(y_pred) > 0
                    else np.nan
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "target": task,
                        "method": method,
                        "n_common_test_videos": len(common),
                        "mae": float(np.mean(np.abs(y_pred - y_true))),
                        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                        "pearson_r": pearson,
                    }
                )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(FIGURE_DIR / "comparison_common_test_metrics.csv", index=False)
    return comparison


def plot_common_test_comparison(comparison):
    x = np.arange(len(TASKS), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    for row, architecture in enumerate(ARCHITECTURES):
        selected_architecture = comparison.loc[
            comparison["architecture"].eq(architecture)
        ]
        counts = (
            selected_architecture.loc[
                selected_architecture["method"].eq("multi_output")
            ]
            .set_index("target")
            .loc[list(TASKS), "n_common_test_videos"]
        )
        labels = [
            f"{TASK_LABELS[target]}\nn={int(counts.loc[target])}" for target in TASKS
        ]
        for column, (metric, ylabel, ylim) in enumerate(
            (
                ("mae", "Video-level MAE", None),
                ("pearson_r", "Video-level Pearson r", (-1.05, 1.05)),
            )
        ):
            axis = axes[row, column]
            for method_index, method in enumerate(("multi_output", "single_task")):
                selected = (
                    selected_architecture.loc[
                        selected_architecture["method"].eq(method)
                    ]
                    .set_index("target")
                    .loc[list(TASKS)]
                )
                axis.bar(
                    x + (method_index - 0.5) * width,
                    selected[metric],
                    width=width,
                    color=METHOD_COLORS[method],
                    alpha=0.9,
                    label=METHOD_LABELS[method],
                )
            axis.axhline(0, color="#666666", linestyle=":", linewidth=0.8)
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.set_xticks(x, labels, rotation=23, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(ARCHITECTURE_LABELS[architecture])
            axis.grid(axis="y", alpha=0.24)
            axis.legend()
    figure.suptitle(
        "Strict comparison on videos assigned to test in both experiments\n"
        "Small intersections, especially lactate, make estimates unstable",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "comparison_common_test_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(output_dir=None, single_output_dir=None):
    global OUTPUT_DIR, SINGLE_OUTPUT_DIR, FIGURE_DIR
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir).resolve()
    if single_output_dir is not None:
        SINGLE_OUTPUT_DIR = Path(single_output_dir).resolve()
    FIGURE_DIR = OUTPUT_DIR / "figures"
    _style()
    _validate_complete(OUTPUT_DIR)
    _validate_complete(SINGLE_OUTPUT_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(OUTPUT_DIR / "history_all.csv")
    multi_metrics = pd.read_csv(OUTPUT_DIR / "metrics_all.csv")
    single_metrics = pd.read_csv(SINGLE_OUTPUT_DIR / "metrics_all.csv")
    plot_training_curves(history)
    plot_test_metrics(multi_metrics)
    plot_split_generalization(multi_metrics)
    plot_predictions(multi_metrics)
    plot_reported_comparison(_reported_comparison(multi_metrics, single_metrics))
    plot_common_test_comparison(_common_test_comparison())
    print(f"Saved 6 figures and 2 comparison tables to {FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--single-output-dir", default=None)
    arguments = parser.parse_args()
    main(arguments.output_dir, arguments.single_output_dir)
