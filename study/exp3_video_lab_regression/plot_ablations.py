"""Compare Exp3 ablations on the intersection of held-out videos."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape
from study.exp2_face_pretrained_head32_regression.plot_results import (
    TASK_LABELS, TASK_UNITS,
)

from .config import TARGETS


VARIANT_LABELS = ("Baseline 16f", "Diverse + 30/40", "Middle 48f")
COLORS = ("#73808A", "#D95F02", "#2878B5", "#278245")


def _test_predictions(output_dir, target, prefix):
    path = output_dir / "runs" / target / "video_predictions.csv"
    data = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
    data = data.loc[data.split.eq("test"),
                    ["hospital_id", "video_id", "y_true", "y_pred"]]
    if data.duplicated(["hospital_id", "video_id"]).any():
        raise ValueError(f"Duplicate test video: {path}")
    return data.rename(columns={"y_true": f"truth_{prefix}",
                                "y_pred": f"pred_{prefix}"})


def plot_comparison(baseline_dir, variant_dirs, labels=VARIANT_LABELS,
                    output_name="comparison"):
    baseline_dir = Path(baseline_dir)
    dirs = (baseline_dir, *map(Path, variant_dirs))
    if len(dirs) != len(labels) or len(dirs) > len(COLORS) or not all(
        (directory / "COMPLETE").is_file()
                                  for directory in dirs):
        raise RuntimeError("All compared Exp3 runs must be complete")
    output = baseline_dir / "ablations" / output_name
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGETS:
        frames = [_test_predictions(directory, target, number)
                  for number, directory in enumerate(dirs)]
        paired = frames[0]
        for frame in frames[1:]:
            paired = paired.merge(frame, on=["hospital_id", "video_id"],
                                  how="inner", validate="one_to_one")
        if paired.empty:
            raise RuntimeError(f"No common test videos for {target}")
        truth = paired.truth_0.to_numpy(float)
        for number, label in enumerate(labels):
            if not np.allclose(truth, paired[f"truth_{number}"], rtol=0, atol=1e-7):
                raise AssertionError(f"Test label mismatch for {target}: {label}")
            prediction = paired[f"pred_{number}"].to_numpy(float)
            correlation = (float(np.corrcoef(truth, prediction)[0, 1])
                           if np.std(truth) and np.std(prediction) else np.nan)
            rows.append({
                "target": target, "variant": label,
                "common_test_videos": len(paired),
                "total_variant_test_videos": len(frames[number]),
                "mae": float(np.mean(np.abs(truth - prediction))),
                "pearson_r": correlation,
            })
    results = pd.DataFrame(rows)
    results.to_csv(output / "common_test_metrics.csv", index=False)
    grid_rows, grid_columns = target_grid_shape(len(TARGETS))
    size = target_grid_figsize(grid_rows, grid_columns)
    for metric, ylabel, filename in (
        ("mae", "Test MAE", "common_test_mae.png"),
        ("pearson_r", "Test Pearson r", "common_test_pearson_r.png"),
    ):
        figure, axes = plt.subplots(grid_rows, grid_columns,
                                    figsize=size, squeeze=False)
        for axis, target in zip(axes.flat, TARGETS):
            values = results.loc[results.target.eq(target)]
            axis.bar(range(len(labels)), values[metric],
                     color=COLORS[:len(labels)], width=0.65)
            axis.set_xticks(range(len(labels)), labels, rotation=22, ha="right")
            count = int(values.common_test_videos.iloc[0])
            axis.set_title(f"{TASK_LABELS[target]} | common n={count}")
            axis.set_ylabel(f"{ylabel} ({TASK_UNITS[target]})"
                            if metric == "mae" else ylabel)
            if metric == "pearson_r":
                axis.set_ylim(-1.05, 1.05)
                axis.axhline(0, linestyle=":", linewidth=0.8, color="#666666")
            axis.grid(axis="y", alpha=0.24)
        figure.suptitle(f"Exp3 ablations: {ylabel} on identical held-out videos",
                        fontsize=15)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
    print(f"[ablation-plots-complete] directory={output}", flush=True)


if __name__ == "__main__":
    from .config import OUTPUT_DIR

    plot_comparison(OUTPUT_DIR, (
        OUTPUT_DIR / "ablations/patient_diverse_schedule_30_40",
        OUTPUT_DIR / "ablations/middle48",
    ))
