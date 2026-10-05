"""Compare all-frame and 20-frame test results on identical videos."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_manifest(directory):
    with (directory / "experiment_manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _test_metrics(directory):
    runs = pd.read_csv(directory / "run_index.csv")
    if not runs["status"].eq("ok").all():
        raise RuntimeError(f"Incomplete runs in {directory}")
    metrics = pd.read_csv(directory / "metrics_all.csv")
    test = metrics.loc[metrics["split"].eq("test")].copy()
    if test.duplicated(["architecture", "target"]).any():
        raise RuntimeError(f"Duplicate test metrics in {directory}")
    return test


def compare(reference_dir, candidate_dir, hours):
    reference = _load_manifest(reference_dir)
    candidate = _load_manifest(candidate_dir)
    if reference["result_variant"] != "20frame" or candidate["result_variant"] != "allframes":
        raise ValueError("Expected 20-frame reference and all-frame candidate")
    if float(reference.get("lab_match_max_delta_hours") or 24) != hours or float(
        candidate.get("lab_match_max_delta_hours") or 24
    ) != hours:
        raise ValueError("Matching windows differ")
    if reference["targets"] != candidate["targets"]:
        raise ValueError("Target lists differ")
    rows = []
    for target in reference["targets"]:
        left = pd.read_csv(reference_dir / "task_records" / f"{target}.csv")
        right = pd.read_csv(candidate_dir / "task_records" / f"{target}.csv")
        columns = ["hospital_id", "video_id", "split", "source_sample_id", "raw_value", "match_delta_h"]
        pd.testing.assert_frame_equal(
            left[columns].sort_values("video_id").reset_index(drop=True),
            right[columns].sort_values("video_id").reset_index(drop=True),
            check_dtype=False,
            rtol=0,
            atol=1e-10,
        )
        base_predictions = pd.read_csv(
            reference_dir / "runs" / "efficientnet_b0" / target / "video_predictions.csv"
        )
        candidate_predictions = pd.read_csv(
            candidate_dir / "runs" / "efficientnet_b0" / target / "video_predictions.csv"
        )
        base_predictions = base_predictions.loc[base_predictions["split"].eq("test")]
        candidate_predictions = candidate_predictions.loc[candidate_predictions["split"].eq("test")]
        keys = ["hospital_id", "video_id"]
        joined = base_predictions.merge(
            candidate_predictions,
            on=keys,
            how="outer",
            suffixes=("_20frame", "_allframes"),
            indicator=True,
            validate="one_to_one",
        )
        if not joined["_merge"].eq("both").all() or not np.allclose(
            joined["y_true_20frame"], joined["y_true_allframes"], rtol=0, atol=1e-10
        ):
            raise RuntimeError(f"Different test videos or labels for {target}")
        if not joined["frame_count_20frame"].eq(20).all():
            raise RuntimeError(f"Invalid 20-frame reference for {target}")
        rows.append({"target": target, "test_videos": len(joined),
                     "allframe_median_frames": float(joined["frame_count_allframes"].median())})
    counts = pd.DataFrame(rows)
    base = _test_metrics(reference_dir)
    full = _test_metrics(candidate_dir)
    columns = ["architecture", "target", "n", "mae", "rmse", "pearson_r", "r2"]
    paired = base[columns].merge(
        full[columns], on=["architecture", "target"], how="outer",
        suffixes=("_20frame", "_allframes"), indicator=True, validate="one_to_one"
    )
    if not paired["_merge"].eq("both").all() or not paired["n_20frame"].eq(
        paired["n_allframes"]
    ).all():
        raise RuntimeError("Test metric rows or sample counts differ")
    paired = paired.drop(columns="_merge").merge(counts, on="target", validate="one_to_one")
    paired["mae_ratio_allframes_to_20frame"] = paired["mae_allframes"] / paired["mae_20frame"]
    paired["pearson_r_difference"] = paired["pearson_r_allframes"] - paired["pearson_r_20frame"]
    paired["r2_difference"] = paired["r2_allframes"] - paired["r2_20frame"]
    figures = candidate_dir / "figures"
    figures.mkdir(exist_ok=True)
    paired.to_csv(candidate_dir / "allframes_vs_20frame_test.csv", index=False)

    paired = paired.sort_values("target", ascending=False)
    labels = [name.replace("_", " ").title() for name in paired["target"]]
    y = np.arange(len(paired))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.8, len(paired) * 0.58)),
                             layout="constrained")
    ratio = paired["mae_ratio_allframes_to_20frame"].to_numpy(float)
    difference = paired["pearson_r_difference"].to_numpy(float)
    axes[0].barh(y, ratio, color=np.where(ratio <= 1, "#267a78", "#b75b43"))
    axes[0].axvline(1, color="#30353b", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Test MAE ratio (all frames / 20 frames)")
    axes[0].set_title("Lower is better")
    axes[1].barh(y, difference, color=np.where(difference >= 0, "#267a78", "#b75b43"))
    axes[1].axvline(0, color="#30353b", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Test Pearson r difference (all frames - 20 frames)")
    axes[1].set_title("Higher is better")
    for axis in axes:
        axis.set_yticks(y, labels)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Frame-count ablation | nearest laboratory result within {hours:g} h")
    fig.savefig(figures / "allframes_vs_20frame_test.png", dpi=180)
    plt.close(fig)
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    result = compare(args.reference_dir, args.candidate_dir, args.hours)
    print(f"[comparison] hours={args.hours:g} targets={len(result)} output={args.candidate_dir}", flush=True)


if __name__ == "__main__":
    main()
