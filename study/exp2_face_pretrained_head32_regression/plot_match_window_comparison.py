"""Compare a shorter lab matching window with the retained 24-hour run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import OUTPUT_DIR, TARGETS
from .plot_results import TASK_LABELS, TASK_UNITS
from .source_data import TARGET_ANALYTES


def _read_run(directory, hours):
    with open(directory / "source_data" / "data_quality_report.json", encoding="utf-8") as handle:
        quality = json.load(handle)
    actual_hours = quality["video_match_policy"]["maximum_delta_hours"]
    if actual_hours != hours:
        raise RuntimeError(f"Expected {hours}h matching in {directory}, found {actual_hours}")
    index = pd.read_csv(directory / "run_index.csv")
    if (len(index) != len(TARGETS) or not index.status.eq("ok").all()
            or set(index.target) != set(TARGETS)):
        raise RuntimeError(f"Incomplete target runs in {directory}")
    metrics = pd.read_csv(directory / "metrics_all.csv")
    test = metrics.loc[
        metrics.architecture.eq("efficientnet_b0") & metrics.split.eq("test")
    ].set_index("target")
    if set(test.index) != set(TARGETS) or test.index.has_duplicates:
        raise RuntimeError(f"Incomplete test metrics in {directory}")
    source = pd.read_csv(
        directory / "source_data" / "base_manifest.csv",
        dtype={"hospital_id": str, "video_id": str},
    ).set_index("video_id", verify_integrity=True)
    return test.loc[list(TARGETS)], source


def compare(reference_dir, candidate_dir, candidate_hours=12.0):
    reference_dir, candidate_dir = Path(reference_dir), Path(candidate_dir)
    suffix = f"{candidate_hours:g}h"
    old, old_source = _read_run(reference_dir, 24.0)
    new, new_source = _read_run(candidate_dir, candidate_hours)
    if not new_source.index.isin(old_source.index).all():
        raise RuntimeError(f"{suffix} source contains videos absent from the 24-hour pool")
    rows, paired_rows = [], []
    for target in TARGETS:
        analyte = TARGET_ANALYTES[target]
        labelled = new_source.loc[new_source[target].notna()]
        previous = old_source.loc[labelled.index]
        columns = (target, f"{analyte}_value", f"{analyte}_lab_time_unix")
        for column in columns:
            if not np.allclose(
                labelled[column].to_numpy(float), previous[column].to_numpy(float),
                rtol=0, atol=1e-8,
            ):
                raise RuntimeError(f"Nearest {target} lab event changed within {suffix}")
        if labelled[f"{analyte}_delta_h"].gt(candidate_hours + 1e-6).any():
            raise RuntimeError(f"Lab/video interval exceeds {suffix} for {target}")

        old_records = pd.read_csv(
            reference_dir / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str},
        )
        new_records = pd.read_csv(
            candidate_dir / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str},
        )
        for records, label in ((old_records, "24h"), (new_records, suffix)):
            if records.video_id.duplicated().any():
                raise RuntimeError(f"Duplicate {label} video for {target}")
            if records.groupby("hospital_id").split.nunique().gt(1).any():
                raise RuntimeError(f"Patient leakage in {label} split for {target}")
        if not set(new_records.video_id).issubset(set(old_records.video_id)):
            raise RuntimeError(f"{suffix} task contains new video for {target}")
        old_test = old_records.loc[old_records.split.eq("test")]
        new_test = new_records.loc[new_records.split.eq("test")]
        if int(old.loc[target, "n"]) != len(old_test) or int(new.loc[target, "n"]) != len(new_test):
            raise RuntimeError(f"Test count differs from task records for {target}")
        rows.append({
            "target": target,
            "unit": TASK_UNITS[target],
            "videos_24h": len(old_records),
            f"videos_{suffix}": len(new_records),
            "test_videos_24h": len(old_test),
            f"test_videos_{suffix}": len(new_test),
            "shared_test_videos": len(set(old_test.video_id) & set(new_test.video_id)),
            "shared_test_patients": len(set(old_test.hospital_id) & set(new_test.hospital_id)),
            "mae_24h": old.loc[target, "mae"],
            f"mae_{suffix}": new.loc[target, "mae"],
            "r2_24h": old.loc[target, "r2"],
            f"r2_{suffix}": new.loc[target, "r2"],
            "pearson_r_24h": old.loc[target, "pearson_r"],
            f"pearson_r_{suffix}": new.loc[target, "pearson_r"],
        })
        predictions = []
        for directory in (reference_dir, candidate_dir):
            frame = pd.read_csv(
                directory / "runs" / "efficientnet_b0" / target
                / "video_predictions.csv",
                dtype={"hospital_id": str, "video_id": str},
            )
            frame = frame.loc[frame.split.eq("test")]
            if frame.video_id.duplicated().any() or not frame.frame_count.eq(20).all():
                raise RuntimeError(f"Invalid test predictions for {target}: {directory}")
            predictions.append(frame[["hospital_id", "video_id", "y_true", "y_pred"]])
        shared = predictions[0].merge(
            predictions[1], on=["hospital_id", "video_id"],
            how="inner", validate="one_to_one", suffixes=("_24h", f"_{suffix}"),
        )
        if len(shared) != rows[-1]["shared_test_videos"] or len(shared) < 3:
            raise RuntimeError(f"Unexpected shared held-out test size for {target}")
        if not np.allclose(shared.y_true_24h, shared[f"y_true_{suffix}"], rtol=0, atol=1e-7):
            raise RuntimeError(f"Different true values on shared test videos for {target}")
        true = shared.y_true_24h.to_numpy(float)
        paired = {"target": target, "shared_test_videos": len(shared)}
        for label in ("24h", suffix):
            predicted = shared[f"y_pred_{label}"].to_numpy(float)
            paired[f"mae_{label}"] = mean_absolute_error(true, predicted)
            paired[f"r2_{label}"] = r2_score(true, predicted)
            paired[f"pearson_r_{label}"] = (
                float(pearsonr(true, predicted).statistic)
                if np.std(true) > 0 and np.std(predicted) > 0 else np.nan
            )
        paired_rows.append(paired)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(candidate_dir / "match_window_comparison.csv", index=False)
    _plot_comparison(comparison, candidate_dir / "figures", suffix)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(candidate_dir / "shared_test_comparison.csv", index=False)
    _plot_shared_test(paired, candidate_dir / "figures", suffix)
    (candidate_dir / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[match-window-comparison-complete] {candidate_dir}", flush=True)


def _plot_comparison(data, figures, suffix):
    figures.mkdir(parents=True, exist_ok=True)
    names = [TASK_LABELS[target] for target in data.target]
    x = np.arange(len(data))
    colors = ("#297F88", "#CB6547")
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for axis, field, ylabel in zip(
        axes,
        ("videos", "r2", "pearson_r"),
        ("Labelled videos", "Test R²", "Test Pearson r"),
    ):
        axis.bar(x - 0.2, data[f"{field}_24h"], width=0.38,
                 color=colors[0], label="24 h")
        axis.bar(x + 0.2, data[f"{field}_{suffix}"], width=0.38,
                 color=colors[1], label=suffix.replace("h", " h"))
        axis.set_ylabel(ylabel)
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x, names, rotation=25, ha="right")
    figure.suptitle(f"Lab/video matching window: 24 h vs {suffix.replace('h', ' h')}", fontsize=15)
    figure.text(
        0.5, 0.015,
        "Patient-disjoint splits were selected independently; test cohorts differ.",
        ha="center", fontsize=9, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.98))
    figure.savefig(figures / "match_window_comparison.png", dpi=180)
    plt.close(figure)

    rows, columns = target_grid_shape(len(data))
    figure, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False,
    )
    for axis, item in zip(axes.flat, data.itertuples(index=False)):
        axis.bar((0, 1), (item.mae_24h, getattr(item, f"mae_{suffix}")),
                 color=colors, width=0.56)
        axis.set_xticks((0, 1), (f"24 h\nn={item.test_videos_24h}",
                                f"{suffix.replace('h', ' h')}\n"
                                f"n={getattr(item, f'test_videos_{suffix}')}"))
        axis.set_title(TASK_LABELS[item.target])
        axis.set_ylabel(f"Test MAE ({item.unit})")
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    for axis in axes.flat[len(data):]:
        axis.axis("off")
    figure.suptitle("Test MAE by laboratory matching window", fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "match_window_mae.png", dpi=180)
    plt.close(figure)


def _plot_shared_test(data, figures, suffix):
    names = [TASK_LABELS[target] for target in data.target]
    x = np.arange(len(data))
    figure, axes = plt.subplots(2, 1, figsize=(15, 7.8), sharex=True)
    for shift, label, color in ((-0.2, "24h", "#297F88"),
                                (0.2, suffix, "#CB6547")):
        axes[0].bar(x + shift, data[f"pearson_r_{label}"], width=0.38,
                    color=color, label=label)
    axes[0].set_ylabel("Pearson r on shared test videos")
    axes[0].legend(frameon=False, ncol=2)
    change = 100 * (data[f"mae_{suffix}"] / data.mae_24h - 1)
    axes[1].bar(x, change, width=0.55,
                color=np.where(change < 0, "#297F88", "#CB6547"))
    axes[1].set_ylabel("MAE change on shared test videos (%)")
    axes[1].set_xticks(x, [f"{name}\nn={count}" for name, count in zip(
        names, data.shared_test_videos
    )], rotation=18, ha="right")
    for axis in axes:
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    figure.suptitle("Same held-out videos under both matching windows", fontsize=15)
    figure.tight_layout()
    figure.savefig(figures / "shared_test_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--candidate-hours", type=float, default=12.0)
    args = parser.parse_args()
    compare(args.reference_dir, args.candidate_dir, args.candidate_hours)
