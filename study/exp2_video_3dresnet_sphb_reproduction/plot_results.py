"""Paper-aligned regression figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_ROOT = (
    REPO_ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame"
)
HISTORY_ROOT = REPO_ROOT / "study/exp2_face_history_head32_regression/outputs/20frame"
ARCHITECTURES = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}


def _face_predictions(root, architecture):
    path = root / "runs" / architecture / "hemoglobin_low" / "video_predictions.csv"
    frame = pd.read_csv(
        path, dtype={"hospital_id": str, "video_id": str}
    ).loc[lambda data: data["split"].eq("test")].copy()
    return frame[["hospital_id", "video_id", "y_true", "y_pred"]]


def _current_video_predictions(output):
    frame = pd.read_csv(
        output / "video_predictions.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    frame = frame.loc[frame["split"].eq("test")].copy()
    frame["y_true"] = frame["y_true_g_dl"] * 10.0
    frame["y_pred"] = frame["y_pred_g_dl"] * 10.0
    return frame[["hospital_id", "video_id", "y_true", "y_pred"]]


def _align_to_current_cohort(current, candidate, label):
    keys = ["hospital_id", "video_id"]
    if current.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise RuntimeError(f"Duplicate test video keys in {label}")
    aligned = current[keys + ["y_true"]].merge(
        candidate,
        on=keys,
        how="left",
        suffixes=("_current", "_candidate"),
        validate="one_to_one",
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        missing = aligned.loc[~aligned["_merge"].eq("both"), "video_id"].tolist()
        raise RuntimeError(f"{label} lacks {len(missing)} current test videos")
    if not np.allclose(
        aligned["y_true_current"], aligned["y_true_candidate"], atol=1e-4
    ):
        raise RuntimeError(f"Hemoglobin labels differ for {label}")
    return aligned.rename(
        columns={"y_true_current": "y_true", "y_pred": "y_pred"}
    )[["hospital_id", "video_id", "y_true", "y_pred"]]


def _comparison_metrics(experiment, architecture, label, predictions):
    truth = predictions["y_true"].to_numpy(np.float64)
    estimate = predictions["y_pred"].to_numpy(np.float64)
    residual = estimate - truth
    total_variance = np.sum((truth - truth.mean()) ** 2)
    return {
        "experiment": experiment,
        "architecture": architecture,
        "label": label,
        "n_test_videos": len(predictions),
        "mae_g_l": float(np.mean(np.abs(residual))),
        "rmse_g_l": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / total_variance),
        "explained_variance": float(1.0 - np.var(residual) / np.var(truth)),
        "pearson_r": float(np.corrcoef(truth, estimate)[0, 1]),
    }


def _plot_comparison(summary, output_path):
    metrics = (
        ("mae_g_l", "MAE (g/L)"),
        ("rmse_g_l", "RMSE (g/L)"),
        ("r2", "R2"),
        ("explained_variance", "Explained variance"),
        ("pearson_r", "Pearson r"),
    )
    colors = ["#4C78A8", "#72A0CF", "#59A14F", "#E15759", "#F28E8E"]
    labels = [
        "Face Head32 | MobileNetV3",
        "Face Head32 | EfficientNet-B0",
        "Video | Residual 3D CNN",
        "Face + history | MobileNetV3",
        "Face + history | EfficientNet-B0",
    ]
    y = np.arange(len(summary))
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    for axis, (metric, title) in zip(axes.flat, metrics):
        values = summary[metric].to_numpy(float)
        bars = axis.barh(y, values, color=colors, height=0.68)
        axis.bar_label(bars, fmt="%.3f", fontsize=9, padding=4)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(f"Common-cohort test {title}")
        axis.axvline(0, color="#666666", linewidth=0.8, linestyle=":")
        axis.grid(axis="x", alpha=0.2)
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * 0.25, abs(upper) * 0.08, 0.05)
        axis.set_xlim(min(0, lower - padding), upper + padding)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05,
        0.70,
        f"Common held-out videos: {int(summary.n_test_videos.iloc[0])}\n\n"
        "Head32: 20 frames/video\n"
        "Residual 3D CNN: centered contiguous 224 frames/video\n\n"
        "All models are recomputed on the current 3D CNN test cohort.\n"
        "Errors are reported in g/L.",
        va="top",
        fontsize=11,
    )
    figure.suptitle("Hemoglobin regression model comparison", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _generate_comparison(output, figures):
    current = _current_video_predictions(output)
    models = []
    for architecture, architecture_label in ARCHITECTURES.items():
        predictions = _align_to_current_cohort(
            current,
            _face_predictions(PRETRAINED_ROOT, architecture),
            f"Pretrained Head32 {architecture_label}",
        )
        models.append((
            "pretrained_head32",
            architecture,
            f"Pretrained Head32\n{architecture_label}",
            predictions,
        ))
    models.append((
        "video_3dresnet",
        "residual_3d_cnn",
        "Video residual 3D CNN",
        current,
    ))
    for architecture, architecture_label in ARCHITECTURES.items():
        predictions = _align_to_current_cohort(
            current,
            _face_predictions(HISTORY_ROOT, architecture),
            f"History Head32 {architecture_label}",
        )
        models.append((
            "history_head32",
            architecture,
            f"History Head32\n{architecture_label}",
            predictions,
        ))
    summary = pd.DataFrame([
        _comparison_metrics(experiment, architecture, label, predictions)
        for experiment, architecture, label, predictions in models
    ])
    if summary["n_test_videos"].nunique() != 1:
        raise AssertionError("Comparison models do not use one common test cohort")
    summary.to_csv(output / "hb_regression_model_comparison.csv", index=False)
    _plot_comparison(summary, figures / "hb_regression_model_comparison.png")
    return summary


def main(output_dir):
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(output / "history.csv")
    predictions = pd.read_csv(output / "video_predictions.csv")
    test = predictions.loc[predictions["split"].eq("test")]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.epoch, history.train_mse_g_dl2, label="Train MSE")
    axes[0].plot(history.epoch, history.val_mse_g_dl2, label="Validation MSE")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE (g/dL)^2"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(history.epoch, history.val_rmse_g_dl, label="Validation RMSE")
    axes[1].plot(history.epoch, history.val_mae_g_dl, label="Validation MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Error (g/dL)"); axes[1].legend(); axes[1].grid(alpha=.2)
    figure.suptitle("Residual 3D CNN training history")
    figure.tight_layout(); figure.savefig(figures / "training_history.png", dpi=180); plt.close(figure)

    truth, prediction = test.y_true_g_dl.to_numpy(), test.y_pred_g_dl.to_numpy()
    difference, mean = prediction - truth, (prediction + truth) / 2
    bias, sd = difference.mean(), difference.std(ddof=1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(truth, prediction, alpha=.7, s=24)
    limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    axes[0].plot(limits, limits, "--", color="black"); axes[0].set_xlabel("True Hb (g/dL)"); axes[0].set_ylabel("Predicted Hb (g/dL)")
    axes[1].scatter(mean, difference, alpha=.7, s=24)
    axes[1].axhline(bias, color="black"); axes[1].axhline(bias + 1.96 * sd, color="red", linestyle="--"); axes[1].axhline(bias - 1.96 * sd, color="red", linestyle="--")
    axes[1].set_xlabel("Mean Hb (g/dL)"); axes[1].set_ylabel("Predicted - true (g/dL)"); axes[1].set_title(f"Bland-Altman: bias={bias:.3f}")
    figure.suptitle("Video-level held-out test results")
    figure.tight_layout(); figure.savefig(figures / "test_regression_and_bland_altman.png", dpi=180); plt.close(figure)
    comparison = _generate_comparison(output, figures)
    print(comparison.drop(columns="label").to_string(index=False), flush=True)
    print(f"Saved 3 figures to {figures}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.output_dir)
