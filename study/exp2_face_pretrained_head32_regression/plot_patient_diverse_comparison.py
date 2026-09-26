"""Compare patient-diverse batches against the unchanged 20-frame baseline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGETS
from .plot_results import TASK_LABELS


def _test_metrics(directory):
    index = pd.read_csv(directory / "run_index.csv")
    if len(index) != len(TARGETS) or not index.status.eq("ok").all():
        raise RuntimeError(f"Incomplete runs: {directory}")
    result = pd.read_csv(directory / "metrics_all.csv")
    result = result.loc[result.split.eq("test")].set_index("target")
    if len(result) != len(TARGETS) or set(result.index) != set(TARGETS):
        raise RuntimeError(f"Incomplete test metrics: {directory}")
    return result.loc[list(TARGETS)]


def _verify_same_test_videos(baseline_dir, candidate_dir):
    for target in TARGETS:
        paths = [
            directory / "runs" / "efficientnet_b0" / target / "video_predictions.csv"
            for directory in (baseline_dir, candidate_dir)
        ]
        frames = []
        for path in paths:
            frame = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
            frame = frame.loc[frame.split.eq("test")].sort_values(
                ["hospital_id", "video_id"]
            ).reset_index(drop=True)
            if frame[["hospital_id", "video_id"]].duplicated().any():
                raise AssertionError(f"Duplicate test video: {path}")
            frames.append(frame)
        for column in ("hospital_id", "video_id", "frame_count"):
            if not frames[0][column].equals(frames[1][column]):
                raise AssertionError(f"Test {column} differs for {target}")
        if not np.allclose(frames[0].y_true, frames[1].y_true, rtol=0, atol=1e-7):
            raise AssertionError(f"Test targets differ for {target}")
        if not frames[0].frame_count.eq(20).all():
            raise AssertionError(f"Unexpected test frame count for {target}")


def plot_comparison(reference_dir, candidate_dir, reference_label="Original batches",
                    candidate_label="Patient-diverse batches", figure_prefix="baseline"):
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    if not (candidate_dir / "COMPLETE").is_file():
        raise RuntimeError(f"Ablation is not complete: {candidate_dir}")
    reference, candidate = _test_metrics(reference_dir), _test_metrics(candidate_dir)
    if not reference.n.equals(candidate.n):
        raise AssertionError("Held-out video counts differ")
    _verify_same_test_videos(reference_dir, candidate_dir)
    figures = candidate_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame({
        "target": TARGETS,
        "test_videos": reference.n.to_numpy(int),
        "reference_mae": reference.mae.to_numpy(float),
        "candidate_mae": candidate.mae.to_numpy(float),
        "reference_pearson_r": reference.pearson_r.to_numpy(float),
        "candidate_pearson_r": candidate.pearson_r.to_numpy(float),
        "reference_r2": reference.r2.to_numpy(float),
        "candidate_r2": candidate.r2.to_numpy(float),
    })
    comparison["mae_change_percent"] = 100 * (
        comparison.candidate_mae / comparison.reference_mae - 1
    )
    comparison["pearson_r_change"] = (
        comparison.candidate_pearson_r - comparison.reference_pearson_r
    )
    comparison["r2_change"] = comparison.candidate_r2 - comparison.reference_r2
    comparison.to_csv(candidate_dir / f"{figure_prefix}_comparison.csv", index=False)

    labels = [TASK_LABELS[target] for target in TARGETS]
    x = np.arange(len(TARGETS))
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for axis, column, title, improved_when_negative in (
        (axes[0], "mae_change_percent", "Test MAE change (%)", True),
        (axes[1], "pearson_r_change", "Test Pearson r change", False),
        (axes[2], "r2_change", "Test R2 change", False),
    ):
        values = comparison[column].to_numpy(float)
        improved = values < 0 if improved_when_negative else values > 0
        axis.bar(x, values, color=np.where(improved, "#278245", "#C45745"))
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    figure.suptitle(
        f"{candidate_label} vs {reference_label} | same held-out videos"
    )
    figure.tight_layout()
    figure.savefig(figures / f"{figure_prefix}_comparison.png", dpi=180)
    plt.close(figure)

    histories = [pd.read_csv(directory / "history_all.csv") for directory in (reference_dir, candidate_dir)]
    rows, columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False,
    )
    for axis, target in zip(axes.flat, TARGETS):
        for history, label, color in zip(
            histories, (reference_label, candidate_label),
            ("#2878B5", "#D05B3E"),
        ):
            selected = history.loc[history.target.eq(target)].sort_values("global_epoch")
            if selected.empty:
                raise RuntimeError(f"Missing history for {target}: {label}")
            axis.plot(selected.global_epoch, selected.val_mae, color=color, label=label)
        axis.set_title(TASK_LABELS[target])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation MAE")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Validation history | same split and training protocol")
    figure.tight_layout()
    figure.savefig(figures / "val_mae_comparison.png", dpi=180)
    plt.close(figure)
    print(f"[comparison-complete] output={figures}", flush=True)


if __name__ == "__main__":
    from .run_ablations import ABLATION_DIR, BASE_DIR

    plot_comparison(BASE_DIR, ABLATION_DIR / "patient_diverse_batches")
