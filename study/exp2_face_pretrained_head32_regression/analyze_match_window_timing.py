"""Compare saved video/laboratory matching distances across 24/12/6h runs."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGETS
from .plot_results import TASK_LABELS


HERE = Path(__file__).resolve().parent
ROOT = HERE / "outputs"
SOURCES = {
    24: ROOT / "20frame",
    12: ROOT / "ablations/lab_match_12h",
    6: ROOT / "ablations/lab_match_6h",
}
DESTINATION = ROOT / "match_window_timing"
EDGES = (0, 3, 6, 12, 24)
BIN_NAMES = ("0-3h", "3-6h", "6-12h", "12-24h")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records():
    parts, hashes = [], {}
    for hours, root in SOURCES.items():
        if hours != 24 and not (root / "COMPLETE").is_file():
            raise RuntimeError(f"Matching-window result is incomplete: {root}")
        index = pd.read_csv(root / "run_index.csv")
        if (len(index) != len(TARGETS) or set(index.target) != set(TARGETS)
                or not index.status.eq("ok").all()):
            raise RuntimeError(f"Incomplete training result: {root}")
        hashes[str(hours)] = {}
        for target in TARGETS:
            path = root / "task_records" / f"{target}.csv"
            frame = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
            delta = frame.match_delta_h.to_numpy(float)
            signed = frame.match_signed_delta_h.to_numpy(float)
            if (frame.video_id.duplicated().any()
                    or frame.groupby("hospital_id").split.nunique().max() != 1
                    or set(frame.split) != {"train", "val", "test"}
                    or not np.isfinite(delta).all()
                    or not np.isfinite(signed).all()
                    or (delta < 0).any() or (delta > hours + 1e-9).any()
                    or not np.allclose(delta, np.abs(signed), rtol=0, atol=1e-6)):
                raise AssertionError(f"Invalid match distances or patient split: {hours}h/{target}")
            frame.insert(0, "target", target)
            frame.insert(0, "window_h", hours)
            parts.append(frame)
            hashes[str(hours)][target] = _sha256(path)
    combined = pd.concat(parts, ignore_index=True)
    for hours in (12, 6):
        for target in TARGETS:
            reference = combined.loc[combined.window_h.eq(24) & combined.target.eq(target)]
            shorter = combined.loc[combined.window_h.eq(hours) & combined.target.eq(target)]
            expected = reference.loc[reference.match_delta_h.le(hours)]
            if set(expected.video_id) != set(shorter.video_id):
                raise AssertionError(f"The {hours}h video cohort is not the 24h time-filtered cohort: {target}")
            comparison = shorter.set_index("video_id").loc[expected.video_id]
            for field in ("hospital_id", "source_sample_id", "binary_label", "raw_value"):
                if not comparison[field].reset_index(drop=True).equals(
                    expected[field].reset_index(drop=True)
                ):
                    raise AssertionError(f"The {hours}h {field} labels differ: {target}")
            if not np.allclose(comparison.match_delta_h, expected.match_delta_h,
                               rtol=0, atol=1e-9):
                raise AssertionError(f"The {hours}h match distances differ: {target}")
    combined["time_bin"] = pd.cut(combined.match_delta_h, EDGES,
                                  labels=BIN_NAMES, right=False,
                                  include_lowest=True).astype(str)
    combined.loc[combined.match_delta_h.eq(24), "time_bin"] = BIN_NAMES[-1]
    if not combined.time_bin.isin(BIN_NAMES).all():
        raise AssertionError("A video-analyte record was not assigned a time bin")
    combined["lab_position"] = np.select(
        (combined.match_signed_delta_h.lt(0), combined.match_signed_delta_h.gt(0)),
        ("before", "after"), default="during"
    )
    return combined, hashes


def _summary(group, hours, target, split="all"):
    row = {
        "window_h": hours, "target": target, "split": split,
        "n_video_analyte_records": len(group),
        "n_unique_videos": group.video_id.nunique(),
        "n_patients": group.hospital_id.nunique(),
        "mean_h": group.match_delta_h.mean(),
        "max_h": group.match_delta_h.max(),
    }
    for q, label in ((0.25, "p25_h"), (0.5, "median_h"), (0.75, "p75_h"),
                     (0.9, "p90_h"), (0.95, "p95_h")):
        row[label] = group.match_delta_h.quantile(q)
    for cutoff in (3, 6, 12):
        row[f"within_{cutoff}h_fraction"] = group.match_delta_h.le(cutoff).mean()
    for position in ("before", "during", "after"):
        row[f"lab_{position}_video_fraction"] = group.lab_position.eq(position).mean()
    return row


def _ecdf(axis, values, label, color):
    values = np.sort(np.asarray(values, dtype=float))
    axis.plot(values, np.arange(1, len(values) + 1) / len(values),
              color=color, label=label, linewidth=1.8)


def plot_distributions(records):
    figures = DESTINATION / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(rows, columns,
                                figsize=target_grid_figsize(rows, columns),
                                squeeze=False, constrained_layout=True)
    colors = {24: "#6B7886", 12: "#278078", 6: "#BD654A"}
    for axis, target in zip(axes.flat, TARGETS):
        counts = []
        for hours in (24, 12, 6):
            group = records.loc[records.target.eq(target) & records.window_h.eq(hours)]
            _ecdf(axis, group.match_delta_h, f"{hours} h", colors[hours])
            counts.append(len(group))
        axis.set_title(f"{TASK_LABELS[target]} | n={counts[0]}/{counts[1]}/{counts[2]}")
        axis.set_xlabel("Lab-to-video-session distance (h)")
        axis.set_ylabel("Cumulative fraction")
        axis.set_xlim(0, 24)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(title="Maximum", fontsize=8)
    figure.savefig(figures / "match_distance_ecdf.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(rows, columns,
                                figsize=target_grid_figsize(rows, columns),
                                squeeze=False, constrained_layout=True)
    bin_colors = ("#347E89", "#65A883", "#D4A253", "#B86255")
    for axis, target in zip(axes.flat, TARGETS):
        for y, hours in enumerate((24, 12, 6)):
            group = records.loc[records.target.eq(target) & records.window_h.eq(hours)]
            fractions = group.time_bin.value_counts(normalize=True).reindex(BIN_NAMES, fill_value=0)
            left = 0.0
            for index, bin_name in enumerate(BIN_NAMES):
                value = float(fractions[bin_name])
                axis.barh(y, value, left=left, color=bin_colors[index], height=0.6,
                          label=bin_name if y == 0 else None)
                left += value
            axis.text(1.01, y, f"n={len(group)}", va="center", fontsize=8)
        axis.set_yticks((0, 1, 2), ("24 h", "12 h", "6 h"))
        axis.invert_yaxis()
        axis.set_xlim(0, 1.28)
        axis.set_xlabel("Fraction of video-analyte records")
        axis.set_title(TASK_LABELS[target])
        axis.grid(axis="x", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  ncol=4, fontsize=8)
    figure.savefig(figures / "match_distance_bins.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    records, hashes = load_records()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    audit_columns = ["window_h", "target", "split", "hospital_id", "video_id",
                     "match_delta_h", "match_signed_delta_h", "time_bin", "lab_position"]
    records[audit_columns].to_csv(DESTINATION / "match_distance_audit.csv", index=False)
    summary = []
    by_split = []
    bin_rows = []
    for hours in (24, 12, 6):
        window = records.loc[records.window_h.eq(hours)]
        summary.append(_summary(window, hours, "ALL_TASK_VIDEO_RECORDS"))
        for target in TARGETS:
            group = window.loc[window.target.eq(target)]
            summary.append(_summary(group, hours, target))
            for split, subset in group.groupby("split"):
                by_split.append(_summary(subset, hours, target, split))
            for bin_name in BIN_NAMES:
                subset = group.loc[group.time_bin.eq(bin_name)]
                bin_rows.append({"window_h": hours, "target": target,
                                 "time_bin": bin_name, "n_videos": len(subset),
                                 "n_patients": subset.hospital_id.nunique(),
                                 "fraction": len(subset) / len(group)})
    pd.DataFrame(summary).to_csv(DESTINATION / "match_distance_summary.csv", index=False)
    pd.DataFrame(by_split).to_csv(DESTINATION / "match_distance_by_split.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(DESTINATION / "match_distance_bins.csv", index=False)
    plot_distributions(records)
    (DESTINATION / "analysis_manifest.json").write_text(json.dumps({
        "source_sha256": hashes,
        "analysis_unit": "one video/target laboratory match; videos repeat across targets",
        "match_distance": "absolute distance from laboratory time to video session [start,end], zero within session",
        "windows_hours": [24, 12, 6],
        "time_bin_edges_hours": list(EDGES),
        "bin_policy": "left-closed/right-open; 24 included in last bin",
        "split_policy": "each window has its own patient-disjoint searched split",
        "records_per_window": {str(h): int(records.window_h.eq(h).sum()) for h in SOURCES},
        "unique_videos_per_window": {
            str(h): int(records.loc[records.window_h.eq(h), "video_id"].nunique())
            for h in SOURCES
        },
    }, indent=2), encoding="utf-8")
    print(f"[match-timing-complete] output={DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
