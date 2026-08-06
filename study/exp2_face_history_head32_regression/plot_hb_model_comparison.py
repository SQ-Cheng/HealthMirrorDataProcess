"""Compare matched-test hemoglobin regression models across three experiments."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_ROOT = REPO_ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame"
HISTORY_ROOT = REPO_ROOT / "study/exp2_face_history_head32_regression/outputs/20frame"
VIDEO_ROOT = REPO_ROOT / "study/exp2_video_3dresnet_sphb_reproduction/outputs"
OUTPUT_ROOT = REPO_ROOT / "study/exp2_face_history_head32_regression/outputs"

ARCHITECTURES = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}


def _face_predictions(root, architecture):
    path = root / "runs" / architecture / "hemoglobin_low" / "video_predictions.csv"
    frame = pd.read_csv(path).loc[lambda data: data["split"].eq("test")].copy()
    return frame[["hospital_id", "video_id", "y_true", "y_pred"]]


def _video_predictions():
    frame = pd.read_csv(VIDEO_ROOT / "video_predictions.csv")
    frame = frame.loc[frame["split"].eq("test")].copy()
    frame["y_true"] = frame["y_true_g_dl"] * 10.0
    frame["y_pred"] = frame["y_pred_g_dl"] * 10.0
    return frame[["hospital_id", "video_id", "y_true", "y_pred"]]


def _validate_alignment(reference, candidate, label):
    keys = ["hospital_id", "video_id"]
    joined = reference.merge(candidate, on=keys, suffixes=("_reference", "_candidate"),
                             how="outer", indicator=True, validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise RuntimeError(f"Test videos do not match for {label}")
    if not np.allclose(joined["y_true_reference"], joined["y_true_candidate"], atol=1e-5):
        raise RuntimeError(f"Hemoglobin test labels do not match for {label}")


def _metrics(experiment, architecture, label, predictions):
    truth = predictions["y_true"].to_numpy(np.float64)
    estimate = predictions["y_pred"].to_numpy(np.float64)
    return {
        "experiment": experiment,
        "architecture": architecture,
        "label": label,
        "n_test_videos": len(predictions),
        "mae_g_l": mean_absolute_error(truth, estimate),
        "rmse_g_l": mean_squared_error(truth, estimate) ** 0.5,
        "r2": r2_score(truth, estimate),
        "explained_variance": explained_variance_score(truth, estimate),
        "pearson_r": np.corrcoef(truth, estimate)[0, 1],
    }


def _plot(summary, output_path):
    metrics = (
        ("mae_g_l", "MAE (g/L)", False),
        ("rmse_g_l", "RMSE (g/L)", False),
        ("r2", "R²", True),
        ("explained_variance", "Explained variance", True),
    )
    colors = ["#4C78A8", "#72A0CF", "#59A14F", "#E15759", "#F28E8E"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(summary))
    labels = summary["label"].tolist()
    for axis, (metric, title, _) in zip(axes.flat, metrics):
        values = summary[metric].to_numpy(float)
        bars = axis.bar(x, values, color=colors, width=0.72)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.set_title(f"Test {title}")
        axis.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
        axis.grid(axis="y", alpha=0.2)
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * 0.2, 0.05)
        axis.set_ylim(min(0, lower - padding), upper + padding)
    figure.suptitle("Hemoglobin regression on the same 185 held-out videos", fontsize=15)
    figure.text(
        0.5, 0.005,
        "Head32 models use 20 frames/video; residual 3D CNN uses 224 frames/video. "
        "All errors are reported in g/L.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    models = []
    for architecture, architecture_label in ARCHITECTURES.items():
        models.append(("pretrained_head32", architecture,
                       f"Pretrained Head32\n{architecture_label}",
                       _face_predictions(PRETRAINED_ROOT, architecture)))
    models.append(("video_3dresnet", "residual_3d_cnn", "Video residual 3D CNN",
                   _video_predictions()))
    for architecture, architecture_label in ARCHITECTURES.items():
        models.append(("history_head32", architecture,
                       f"History Head32\n{architecture_label}",
                       _face_predictions(HISTORY_ROOT, architecture)))

    reference = models[0][3]
    for experiment, architecture, label, predictions in models:
        _validate_alignment(reference, predictions, label)

    summary = pd.DataFrame([
        _metrics(experiment, architecture, label, predictions)
        for experiment, architecture, label, predictions in models
    ])
    figure_dir = OUTPUT_ROOT / "figures"
    table_dir = OUTPUT_ROOT / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "hb_regression_model_comparison.csv"
    figure_path = figure_dir / "hb_regression_model_comparison.png"
    summary.to_csv(table_path, index=False)
    _plot(summary, figure_path)
    print(summary.drop(columns="label").to_string(index=False))
    print(f"Saved table to {table_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
