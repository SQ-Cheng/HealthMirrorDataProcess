"""Audit and summarize the main Exp6 face/laboratory pair timing."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, TARGETS
from .plot_results import DISPLAY


DESTINATION = OUTPUT_DIR / "timing_analysis"
METRICS = (
    "video_interval_h", "lab_interval_h", "abs_interval_mismatch_h",
    "first_match_delta_h", "second_match_delta_h", "worst_match_delta_h",
    "first_lab_minus_video_midpoint_h", "second_lab_minus_video_midpoint_h",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate():
    index = pd.read_csv(OUTPUT_DIR / "run_index.csv")
    if (len(index) != len(TARGETS) or set(index.target) != set(TARGETS)
            or not index.status.eq("ok").all()):
        raise RuntimeError("The nine-task main Exp6 run is incomplete")
    video_path = OUTPUT_DIR / "source_data/video_summary.csv"
    videos = pd.read_csv(video_path, dtype={"video_id": str, "hospital_id": str})
    if videos.video_id.duplicated().any():
        raise AssertionError("Duplicate video identifiers in source summary")
    videos = videos.set_index("video_id")
    base_path = OUTPUT_DIR / "source_data/base_manifest.csv"
    base = pd.read_csv(base_path, dtype={"video_id": str, "hospital_id": str})
    if base.video_id.duplicated().any():
        raise AssertionError("Duplicate videos in source base manifest")
    base = base.set_index("video_id")
    sources = {"video_summary": _sha256(video_path),
               "base_manifest": _sha256(base_path), "task_records": {}}
    parts = []
    for target in TARGETS:
        path = OUTPUT_DIR / "task_records" / f"{target}.csv"
        frame = pd.read_csv(path, dtype={"hospital_id": str,
                                         "first_video_id": str, "second_video_id": str})
        if (frame.pair_id.duplicated().any() or not frame.target.eq(target).all()
                or frame.groupby("hospital_id").split.nunique().max() != 1):
            raise AssertionError(f"Invalid task records: {target}")
        sources["task_records"][target] = _sha256(path)
        parts.append(frame)
    pairs = pd.concat(parts, ignore_index=True)
    if not pairs.split.isin(("train", "val", "test")).all():
        raise AssertionError("Unknown split in Exp6 records")

    for side in ("first", "second"):
        ids = pairs[f"{side}_video_id"]
        if not ids.isin(videos.index).all() or not ids.isin(base.index).all():
            raise AssertionError(f"Missing {side} video in source summary")
        video_hospital_ids = ids.map(videos.hospital_id)
        if not video_hospital_ids.eq(pairs.hospital_id).all():
            raise AssertionError(f"{side} video/patient identity mismatch")
        start = ids.map(videos.capture_start_unix).to_numpy(float)
        end = ids.map(videos.capture_end_unix).to_numpy(float)
        midpoint = ids.map(videos.capture_time_unix).to_numpy(float)
        lab = pairs[f"{side}_lab_time_unix"].to_numpy(float)
        saved_midpoint = pairs[f"{side}_video_time_unix"].to_numpy(float)
        saved_match = pairs[f"{side}_match_delta_h"].to_numpy(float)
        if (not np.isfinite(start).all() or not np.isfinite(end).all()
                or not np.isfinite(lab).all() or (end < start).any()
                or not np.allclose(midpoint, saved_midpoint, rtol=0, atol=1e-5)):
            raise AssertionError(f"Invalid {side} video/session time")
        recomputed_match = np.maximum.reduce((start - lab, lab - end,
                                              np.zeros(len(pairs)))) / 3600
        if not np.allclose(saved_match, recomputed_match, rtol=0, atol=1e-6):
            raise AssertionError(f"{side} lab/video matching distance differs from session bounds")
        pairs[f"{side}_lab_minus_video_midpoint_h"] = (lab - midpoint) / 3600
        pairs[f"{side}_lab_position"] = np.select(
            (lab < start, lab > end), ("before", "after"), default="during"
        )
        pairs[f"{side}_video_duration_min"] = (end - start) / 60
        pairs[f"{side}_admission_unix"] = ids.map(base.admission_unix).to_numpy(float)
        pairs[f"{side}_discharge_unix"] = ids.map(base.discharge_unix).to_numpy(float)
        if not np.isfinite(pairs[f"{side}_admission_unix"]).all():
            raise AssertionError(f"Missing {side} hospitalization episode")

    computed_video_interval = (
        pairs.second_video_time_unix - pairs.first_video_time_unix
    ) / 3600
    computed_lab_interval = (
        pairs.second_lab_time_unix - pairs.first_lab_time_unix
    ) / 3600
    if (not (computed_video_interval > 0).all()
            or not (computed_lab_interval > 0).all()
            or not np.allclose(pairs.video_interval_h, computed_video_interval, rtol=0, atol=1e-6)
            or not np.allclose(pairs.lab_interval_h, computed_lab_interval, rtol=0, atol=1e-6)
            or not np.allclose(pairs.raw_delta, pairs.second_value - pairs.first_value,
                               rtol=0, atol=1e-10)):
        raise AssertionError("Pair chronology, saved intervals, or delta labels are inconsistent")
    pairs["interval_mismatch_h"] = pairs.video_interval_h - pairs.lab_interval_h
    pairs["abs_interval_mismatch_h"] = pairs.interval_mismatch_h.abs()
    pairs["worst_match_delta_h"] = pairs[[
        "first_match_delta_h", "second_match_delta_h"
    ]].max(axis=1)
    pairs["same_hospital_episode"] = (
        pairs.first_admission_unix.eq(pairs.second_admission_unix)
        & pairs.first_discharge_unix.eq(pairs.second_discharge_unix)
    )
    if not pairs.worst_match_delta_h.le(24 + 1e-6).all():
        raise AssertionError("Main Exp6 includes a pair matched beyond 24 hours")
    if not np.allclose(
        pairs.interval_mismatch_h,
        pairs.first_lab_minus_video_midpoint_h - pairs.second_lab_minus_video_midpoint_h,
        rtol=0, atol=1e-6,
    ):
        raise AssertionError("Pair interval mismatch disagrees with endpoint offsets")
    return pairs, sources


def _summary(group, target, split="all"):
    video_pairs = group[["hospital_id", "first_video_id", "second_video_id"]].drop_duplicates()
    row = {
        "target": target, "split": split, "n_task_pairs": len(group),
        "n_patients": group.hospital_id.nunique(),
        "n_unique_video_pairs": len(video_pairs),
        "n_cross_episode_pairs": int((~group.same_hospital_episode).sum()),
    }
    for field in METRICS:
        values = group[field]
        for quantile, label in ((0.25, "p25"), (0.5, "median"), (0.75, "p75"),
                                (0.9, "p90"), (0.95, "p95")):
            row[f"{field}_{label}"] = values.quantile(quantile)
        row[f"{field}_max"] = values.max()
    for hours in (3, 6, 12, 24):
        row[f"both_matches_within_{hours}h_n"] = int(group.worst_match_delta_h.le(hours).sum())
        row[f"both_matches_within_{hours}h_fraction"] = group.worst_match_delta_h.le(hours).mean()
        row[f"interval_mismatch_within_{hours}h_fraction"] = group.abs_interval_mismatch_h.le(hours).mean()
    for side in ("first", "second"):
        for position in ("before", "during", "after"):
            row[f"{side}_lab_{position}_video_fraction"] = group[f"{side}_lab_position"].eq(position).mean()
    return row


def _ecdf(axis, values, label, color):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    axis.plot(sorted_values, np.arange(1, len(sorted_values) + 1) / len(sorted_values),
              color=color, linewidth=1.8, label=label)


def plot_ecdfs(pairs):
    figures = DESTINATION / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("intervals", ("video_interval_h", "lab_interval_h"),
         ("Video interval", "Lab interval"), "Hours between observations", "log"),
        ("matching_distance", ("first_match_delta_h", "second_match_delta_h"),
         ("Earlier face/lab", "Later face/lab"), "Hours to video session", "linear"),
        ("interval_mismatch", ("abs_interval_mismatch_h",),
         ("|Video interval - lab interval|",), "Absolute interval difference (h)", "linear"),
    )
    for name, fields, labels, xlabel, scale in specifications:
        figure, axes = plt.subplots(3, 3, figsize=(16.5, 12.5), constrained_layout=True)
        for axis, target in zip(axes.flat, TARGETS):
            group = pairs.loc[pairs.target.eq(target)]
            for field, label, color in zip(fields, labels, ("#287F88", "#C25E49")):
                _ecdf(axis, group[field], label, color)
            axis.set_title(f"{DISPLAY.get(target, target)} | n={len(group)}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Cumulative fraction")
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.2)
            if scale == "log":
                axis.set_xscale("log")
                axis.set_xlim(left=0.1)
                axis.axvline(24, color="#666666", linestyle=":", linewidth=1)
            else:
                for cutoff in (6, 12):
                    axis.axvline(cutoff, color="#888888", linestyle=":", linewidth=1)
                axis.set_xlim(left=0, right=24 if name == "matching_distance" else 48)
        axes.flat[0].legend(fontsize=8, loc="lower right")
        figure.savefig(figures / f"{name}_ecdf.png", dpi=180)
        plt.close(figure)


def main():
    pairs, sources = load_and_validate()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(DESTINATION / "pair_timing_audit.csv", index=False)
    pairs.loc[~pairs.same_hospital_episode].to_csv(
        DESTINATION / "cross_episode_pairs.csv", index=False
    )
    unique_video_pairs = pairs.drop_duplicates(
        ["hospital_id", "first_video_id", "second_video_id"]
    )[["hospital_id", "first_video_id", "second_video_id", "video_interval_h",
       "same_hospital_episode"]]
    unique_video_pairs.to_csv(DESTINATION / "unique_video_pairs.csv", index=False)
    summary = [_summary(pairs, "ALL_TASK_PAIRS")]
    summary.extend(_summary(pairs.loc[pairs.target.eq(target)], target) for target in TARGETS)
    pd.DataFrame(summary).to_csv(DESTINATION / "timing_by_target.csv", index=False)
    split_rows = [
        _summary(group, target, split)
        for target in TARGETS
        for split, group in pairs.loc[pairs.target.eq(target)].groupby("split")
    ]
    pd.DataFrame(split_rows).to_csv(DESTINATION / "timing_by_target_split.csv", index=False)
    plot_ecdfs(pairs)
    (DESTINATION / "analysis_manifest.json").write_text(json.dumps({
        "source": "main Exp6 24h run only; no retraining",
        "source_sha256": sources,
        "match_distance": "nearest distance from lab timestamp to video session [start,end], zero within session",
        "video_interval": "difference between video session midpoints",
        "lab_interval": "difference between matched laboratory timestamps",
        "interval_mismatch": "video_interval minus lab_interval",
        "analysis_unit": "one target-specific chronological lab/face pair; video pairs repeat across analytes",
        "n_task_pairs": len(pairs),
        "n_patients": pairs.hospital_id.nunique(),
        "n_unique_video_pairs": len(pairs[[
            "hospital_id", "first_video_id", "second_video_id"
        ]].drop_duplicates()),
        "unique_video_pair_interval_median_h": float(unique_video_pairs.video_interval_h.median()),
        "n_cross_episode_task_pairs": int((~pairs.same_hospital_episode).sum()),
    }, indent=2), encoding="utf-8")
    print(f"[timing-analysis-complete] task_pairs={len(pairs)} "
          f"patients={pairs.hospital_id.nunique()} output={DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
