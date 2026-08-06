"""Generate Mamba regression figures and controlled comparison summaries."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
MAMBA_OUTPUT = EXP_DIR / "outputs"
ABLATION_OUTPUT = EXP_DIR.parent / "exp2_video_only_mamba_ablation" / "outputs"
HEAD32_OUTPUT = (
    EXP_DIR.parent / "exp2_face_pretrained_head32_regression" / "outputs"
)
FIGURE_DIR = MAMBA_OUTPUT / "figures"

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
    "high_blood_pressure": "High blood\npressure",
    "lactate_high": "Lactate high",
}
METHODS = (
    "video_ecg_mamba",
    "mobilenet_v3_small",
    "efficientnet_b0",
)
METHOD_LABELS = {
    "video_ecg_mamba": "Video + ECG Mamba",
    "video_only_mamba": "Video-only Mamba",
    "mobilenet_v3_small": "MobileNetV3-Small Head32",
    "efficientnet_b0": "EfficientNet-B0 Head32",
}
COLORS = {
    "video_ecg_mamba": "#197278",
    "video_only_mamba": "#6B7280",
    "mobilenet_v3_small": "#3B82F6",
    "efficientnet_b0": "#E07A5F",
}
SPLIT_COLORS = {"train": "#3B82F6", "val": "#E07A5F", "test": "#197278"}


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def _require_complete(output_dir, name, architectures=None):
    path = output_dir / "run_index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{name} run index is missing: {path}")
    frame = pd.read_csv(path)
    keys = ["target"] if architectures is None else ["architecture", "target"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} run index contains duplicate job keys")
    if architectures is None:
        expected = set(TASKS)
        successful = set(frame.loc[frame["status"].eq("ok"), "target"])
    else:
        expected = {
            (architecture, target)
            for architecture in architectures
            for target in TASKS
        }
        successful = set(
            frame.loc[frame["status"].eq("ok"), keys].itertuples(
                index=False, name=None
            )
        )
    if successful != expected or frame["status"].ne("ok").any():
        raise RuntimeError(
            f"{name} is incomplete: successful={sorted(successful)}"
        )


def _ordered(frame, index_columns):
    indexed = frame.set_index(index_columns)
    expected = pd.MultiIndex.from_product(
        [
            METHODS if index_columns[0] == "method" else TASKS,
            TASKS if index_columns[0] == "method" else ("train", "val", "test"),
        ],
        names=index_columns,
    )
    missing = expected.difference(indexed.index)
    if len(missing):
        raise ValueError(f"Missing comparison rows: {missing.tolist()}")
    return indexed.loc[expected].reset_index()


def load_results():
    _require_complete(MAMBA_OUTPUT, "video+ECG Mamba")
    _require_complete(ABLATION_OUTPUT, "video-only Mamba")
    _require_complete(
        HEAD32_OUTPUT,
        "Head32 regression",
        architectures=("mobilenet_v3_small", "efficientnet_b0"),
    )

    mamba = pd.read_csv(MAMBA_OUTPUT / "metrics_all.csv")
    ablation = pd.read_csv(ABLATION_OUTPUT / "metrics_all.csv")
    head32 = pd.read_csv(HEAD32_OUTPUT / "metrics_all.csv")
    for name, frame in (("Mamba", mamba), ("ablation", ablation)):
        duplicates = frame.duplicated(["target", "split"])
        if duplicates.any():
            raise ValueError(f"{name} metrics contain duplicate task/split rows")
    if head32.duplicated(["architecture", "target", "split"]).any():
        raise ValueError("Head32 metrics contain duplicate rows")

    test_rows = []
    mamba_test = mamba.loc[mamba["split"].eq("test")]
    for row in mamba_test.itertuples(index=False):
        test_rows.append(
            {
                "method": "video_ecg_mamba",
                "target": row.target,
                "n": int(row.n_videos),
                "mae": row.mae,
                "rmse": row.rmse,
                "r2": row.r2,
                "pearson_r": row.pearson_r,
                "spearman_r": row.spearman_r,
                "sign_roc_auc": row.sign_roc_auc,
                "sign_balanced_accuracy": row.sign_balanced_accuracy,
                "test_split_relationship": "independent_from_head32",
            }
        )
    for row in head32.loc[head32["split"].eq("test")].itertuples(index=False):
        test_rows.append(
            {
                "method": row.architecture,
                "target": row.target,
                "n": int(row.n),
                "mae": row.mae,
                "rmse": row.rmse,
                "r2": row.r2,
                "pearson_r": row.pearson_r,
                "spearman_r": row.spearman_r,
                "sign_roc_auc": row.sign_roc_auc,
                "sign_balanced_accuracy": row.sign_balanced_accuracy,
                "test_split_relationship": "independent_from_mamba",
            }
        )
    comparison = _ordered(
        pd.DataFrame(test_rows), ["method", "target"]
    )
    return mamba, ablation, comparison


def paired_ablation_summary():
    rows = []
    rng = np.random.default_rng(20260728)
    for target in TASKS:
        parent = pd.read_csv(
            MAMBA_OUTPUT / "runs" / target / "video_predictions.csv",
            dtype={"video_id": str, "hospital_id": str},
        )
        ablation = pd.read_csv(
            ABLATION_OUTPUT / "runs" / target / "video_predictions.csv",
            dtype={"video_id": str, "hospital_id": str},
        )
        parent = parent.loc[parent["split"].eq("test")].copy()
        ablation = ablation.loc[ablation["split"].eq("test")].copy()
        paired = parent.merge(
            ablation,
            on=["video_id", "hospital_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("_video_ecg", "_video_only"),
        )
        if len(paired) != len(parent) or len(paired) != len(ablation):
            raise ValueError(f"Unpaired Mamba test videos for {target}")
        if not np.allclose(
            paired["target_score_video_ecg"],
            paired["target_score_video_only"],
            atol=1e-7,
        ):
            raise ValueError(f"Target scores differ in paired ablation for {target}")
        parent_error = np.abs(
            paired["prediction_video_ecg"]
            - paired["target_score_video_ecg"]
        ).to_numpy(np.float64)
        ablation_error = np.abs(
            paired["prediction_video_only"]
            - paired["target_score_video_only"]
        ).to_numpy(np.float64)
        difference = parent_error - ablation_error
        bootstrap = difference[
            rng.integers(0, len(difference), size=(5000, len(difference)))
        ].mean(axis=1)
        rows.append(
            {
                "target": target,
                "n_paired_test_videos": int(len(paired)),
                "video_ecg_mae": float(parent_error.mean()),
                "video_only_mae": float(ablation_error.mean()),
                "mean_absolute_error_difference_ecg_minus_video_only": float(
                    difference.mean()
                ),
                "bootstrap_95ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(bootstrap, 0.975)),
                "fraction_video_ecg_lower_error": float(
                    np.mean(parent_error < ablation_error)
                ),
                "fraction_equal_error": float(
                    np.mean(np.isclose(parent_error, ablation_error, atol=1e-12))
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_training_curves(history, metrics):
    selected_epochs = (
        metrics.loc[metrics["split"].eq("test")]
        .set_index("target")["selected_epoch"]
        .to_dict()
    )
    figure, axes = plt.subplots(5, 2, figsize=(15, 18), squeeze=False)
    for row, target in enumerate(TASKS):
        selected = history.loc[history["target"].eq(target)].sort_values("epoch")
        epochs = selected["epoch"].to_numpy()
        chosen = int(selected_epochs[target])

        loss_axis = axes[row, 0]
        loss_axis.plot(
            epochs, selected["train_loss"], color="#3B82F6", label="Train loss"
        )
        loss_axis.plot(
            epochs, selected["val_loss"], color="#E07A5F", label="Validation loss"
        )
        loss_axis.axvline(
            chosen, color="#197278", linestyle="--", linewidth=1, label="Selected"
        )
        loss_axis.set_ylabel("SmoothL1 loss")
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_title(f"{TASK_LABELS[target]} | optimization")
        loss_axis.grid(axis="y", alpha=0.22)
        loss_axis.legend()

        metric_axis = axes[row, 1]
        metric_axis.plot(
            epochs, selected["train_mae"], color="#3B82F6", label="Train MAE"
        )
        metric_axis.plot(
            epochs, selected["val_mae"], color="#E07A5F", label="Validation MAE"
        )
        metric_axis.axvline(chosen, color="#197278", linestyle="--", linewidth=1)
        metric_axis.set_ylabel("Video-level MAE")
        metric_axis.set_xlabel("Epoch")
        correlation_axis = metric_axis.twinx()
        correlation_axis.plot(
            epochs,
            selected["val_pearson_r"],
            color="#59A14F",
            linewidth=1.1,
            label="Validation Pearson r",
        )
        correlation_axis.set_ylim(-1.02, 1.02)
        correlation_axis.set_ylabel("Pearson r")
        metric_axis.set_title(f"{TASK_LABELS[target]} | generalization")
        metric_axis.grid(axis="y", alpha=0.22)
        handles_a, labels_a = metric_axis.get_legend_handles_labels()
        handles_b, labels_b = correlation_axis.get_legend_handles_labels()
        metric_axis.legend(handles_a + handles_b, labels_a + labels_b)
    figure.suptitle(
        "Native video + 512 Hz ECG Mamba: training history",
        fontsize=15,
        y=1.002,
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mamba_test_metrics(metrics):
    test = metrics.loc[metrics["split"].eq("test")].set_index("target").loc[
        list(TASKS)
    ]
    x = np.arange(len(TASKS))
    labels = [TASK_LABELS[target] for target in TASKS]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    specs = (
        (axes[0, 0], ("mae", "rmse"), ("MAE", "RMSE"), "Prediction error"),
        (
            axes[0, 1],
            ("pearson_r", "spearman_r"),
            ("Pearson r", "Spearman r"),
            "Correlation",
        ),
        (axes[1, 0], ("r2",), ("R2",), "Coefficient of determination"),
        (
            axes[1, 1],
            ("sign_roc_auc", "sign_balanced_accuracy"),
            ("Sign ROC-AUC", "Sign bACC"),
            "Zero-boundary diagnostics",
        ),
    )
    metric_colors = ("#197278", "#E07A5F")
    for axis, columns, names, ylabel in specs:
        width = 0.72 / len(columns)
        for index, (column, name) in enumerate(zip(columns, names)):
            positions = x + (index - (len(columns) - 1) / 2) * width
            axis.bar(
                positions,
                test[column],
                width=width,
                color=metric_colors[index],
                alpha=0.88,
                label=name,
            )
        baseline = 0.5 if "sign_roc_auc" in columns else 0.0
        axis.axhline(
            baseline, color="#555555", linestyle="--", linewidth=0.9
        )
        if "sign_roc_auc" in columns:
            axis.set_ylim(0, 1.04)
        elif "pearson_r" in columns:
            axis.set_ylim(-1.02, 1.02)
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.24)
        axis.legend()
    figure.suptitle(
        "Native video + 512 Hz ECG Mamba: video-level test performance",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "test_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_predictions(metrics):
    test_metrics = metrics.loc[metrics["split"].eq("test")].set_index("target")
    figure, axes = plt.subplots(2, 3, figsize=(16, 10.5), squeeze=False)
    for axis, target in zip(axes.flat, TASKS):
        predictions = pd.read_csv(
            MAMBA_OUTPUT / "runs" / target / "video_predictions.csv"
        )
        predictions = predictions.loc[predictions["split"].eq("test")]
        y_true = predictions["target_score"].to_numpy(np.float64)
        y_pred = predictions["prediction"].to_numpy(np.float64)
        normal = y_true < 0
        axis.scatter(
            y_true[normal],
            y_pred[normal],
            s=28,
            alpha=0.72,
            color="#3B82F6",
            edgecolors="none",
            label="Normal side",
        )
        axis.scatter(
            y_true[~normal],
            y_pred[~normal],
            s=28,
            alpha=0.74,
            color="#E07A5F",
            edgecolors="none",
            label="Abnormal/boundary",
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
        row = test_metrics.loc[target]
        axis.set_title(
            f"{TASK_LABELS[target]}\n"
            f"n={int(row.n_videos)}, MAE={row.mae:.3f}, r={row.pearson_r:.3f}"
        )
        axis.grid(alpha=0.18)
        axis.legend(loc="best")
    axes[1, 2].axis("off")
    figure.suptitle(
        "Native video + 512 Hz ECG Mamba: test predictions",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "test_predicted_vs_true.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_paired_ablation(mamba, ablation, paired):
    parent = mamba.loc[mamba["split"].eq("test")].set_index("target").loc[
        list(TASKS)
    ]
    video_only = ablation.loc[
        ablation["split"].eq("test")
    ].set_index("target").loc[list(TASKS)]
    paired = paired.set_index("target").loc[list(TASKS)]
    x = np.arange(len(TASKS))
    width = 0.36
    labels = [TASK_LABELS[target] for target in TASKS]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))

    for offset, (method, values) in enumerate(
        (("video_ecg_mamba", parent), ("video_only_mamba", video_only))
    ):
        axes[0, 0].bar(
            x + (offset - 0.5) * width,
            values["mae"],
            width=width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[0, 1].bar(
            x + (offset - 0.5) * width,
            values["pearson_r"],
            width=width,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0, 0].set_ylabel("Test MAE")
    axes[0, 1].set_ylabel("Test Pearson r")
    axes[0, 1].set_ylim(-1.02, 1.02)
    axes[0, 1].axhline(0, color="#555555", linestyle=":", linewidth=0.9)
    for axis in axes[0]:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.24)
        axis.legend()

    delta = paired[
        "mean_absolute_error_difference_ecg_minus_video_only"
    ].to_numpy()
    lower = delta - paired["bootstrap_95ci_low"].to_numpy()
    upper = paired["bootstrap_95ci_high"].to_numpy() - delta
    axes[1, 0].errorbar(
        x,
        delta,
        yerr=np.vstack((lower, upper)),
        fmt="o",
        markersize=7,
        capsize=5,
        color=COLORS["video_ecg_mamba"],
        ecolor="#4B5563",
    )
    axes[1, 0].axhline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Paired MAE difference\n(Video+ECG minus video-only)")
    axes[1, 0].set_title("Negative values favor adding ECG; 95% bootstrap CI")
    axes[1, 0].grid(axis="y", alpha=0.24)

    axes[1, 1].bar(
        x,
        paired["fraction_video_ecg_lower_error"],
        width=0.62,
        color=COLORS["video_ecg_mamba"],
    )
    axes[1, 1].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[1, 1].set_ylim(0, 1.04)
    axes[1, 1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1, 1].set_ylabel("Fraction of paired test videos")
    axes[1, 1].set_title("Videos with lower absolute error after adding ECG")
    axes[1, 1].grid(axis="y", alpha=0.24)
    figure.suptitle(
        "Controlled Mamba ablation: identical test videos and splits",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "paired_ecg_ablation.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_head32_comparison(comparison):
    indexed = comparison.set_index(["method", "target"])
    x = np.arange(len(TASKS))
    width = 0.24
    labels = [TASK_LABELS[target] for target in TASKS]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    specs = (
        (axes[0, 0], "mae", "Test MAE", None),
        (axes[0, 1], "rmse", "Test RMSE", None),
        (axes[1, 0], "pearson_r", "Test Pearson r", (-1.02, 1.02)),
        (axes[1, 1], "spearman_r", "Test Spearman r", (-1.02, 1.02)),
    )
    for axis, metric, ylabel, ylim in specs:
        for method_index, method in enumerate(METHODS):
            values = indexed.loc[(method, list(TASKS)), metric].to_numpy()
            positions = x + (method_index - 1) * width
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            if metric == "mae":
                counts = indexed.loc[(method, list(TASKS)), "n"].to_numpy()
                for bar, count in zip(bars, counts):
                    axis.annotate(
                        f"n={int(count)}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=90,
                    )
        axis.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.24)
        axis.legend()
    figure.suptitle(
        "Mamba vs Head32 pretrained regressors: reported video-level test metrics",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.005,
        "Cross-experiment comparison uses each experiment's own patient-level "
        "test split; bars are not paired. Mamba also applies the 60 ms ECG "
        "quality filter.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(
        FIGURE_DIR / "head32_regression_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    _style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    mamba, ablation, comparison = load_results()
    history = pd.read_csv(MAMBA_OUTPUT / "history_all.csv")
    paired = paired_ablation_summary()
    comparison.to_csv(
        MAMBA_OUTPUT / "head32_test_comparison.csv", index=False
    )
    paired.to_csv(MAMBA_OUTPUT / "paired_ablation_summary.csv", index=False)
    plot_training_curves(history, mamba)
    plot_mamba_test_metrics(mamba)
    plot_predictions(mamba)
    plot_paired_ablation(mamba, ablation, paired)
    plot_head32_comparison(comparison)
    print(f"Saved 5 figures to {FIGURE_DIR}")
    print(f"Saved comparison tables to {MAMBA_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
