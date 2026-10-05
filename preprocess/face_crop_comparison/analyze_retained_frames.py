"""Summarize retained-frame geometry from saved CSVs; no video decoding."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BRANCHES = {
    "kalman_aligned": "aligned_eligible",
    "no_kalman_no_alignment": "unfiltered_direct_eligible",
}
METRICS = (
    "confidence", "outside_area_percent", "roi_area_percent", "crop_width",
    "crop_height", "scale_x", "scale_y", "maximum_scale", "padding_percent",
)


def retained_mask(series):
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"Invalid eligibility values in {series.name}")
    return values.isin(["true", "1"])


def load_metrics(output):
    selected = pd.read_csv(output / "tables/selected_videos.csv").set_index("video_id")
    frames, total = [], 0
    for video_id, source in selected.iterrows():
        table = pd.read_csv(output / f"tables/frames/{video_id}.csv")
        if len(table) != source.source_frames or table.source_frame_index.duplicated().any():
            raise ValueError(f"Frame coverage mismatch: {video_id}")
        total += len(table)
        for branch, flag in BRANCHES.items():
            rows = table.loc[retained_mask(table[flag])]
            if rows.empty:
                continue
            width = rows.crop_width if branch == "kalman_aligned" else rows.unfiltered_crop_x2 - rows.unfiltered_crop_x1
            height = rows.crop_height if branch == "kalman_aligned" else rows.unfiltered_crop_y2 - rows.unfiltered_crop_y1
            metrics = pd.DataFrame({
                "confidence": rows.confidence,
                "outside_area_percent": rows.expanded_box_outside_area_fraction * 100,
                "roi_area_percent": width * height / (source.source_width * source.source_height) * 100,
                "crop_width": width, "crop_height": height,
                "scale_x": 224 / width, "scale_y": 224 / height,
                "maximum_scale": np.maximum(224 / width, 224 / height),
                "padding_percent": rows.padding_fraction * 100 if branch == "kalman_aligned" else 0.,
                "video_id": video_id, "branch": branch,
            })
            if not np.isfinite(metrics[list(METRICS)].to_numpy()).all() or (width <= 0).any() or (height <= 0).any():
                raise ValueError(f"Invalid retained geometry: {video_id}, {branch}")
            if (metrics.confidence < .8 - 1e-9).any() or (metrics.outside_area_percent > 10 + 1e-9).any():
                raise ValueError(f"Retained frame violates quality gate: {video_id}")
            frames.append(metrics)
    return pd.concat(frames, ignore_index=True), total


def summarize(data, total, output):
    summaries, bins, videos = [], [], []
    bin_specs = {
        "confidence": ([.8, .85, .9, .95, 1.], ["0.80-0.85", "0.85-0.90", "0.90-0.95", "0.95-1.00"]),
        "outside_area_percent": ([-np.inf, 0, 2, 5, 10.00000001], ["0%", "(0,2]%", "(2,5]%", "(5,10]%"]),
        "maximum_scale": ([-np.inf, 1, 1.25, 1.5, 2, np.inf], ["<=1 (no enlargement)", "(1,1.25]", "(1.25,1.5]", "(1.5,2]", ">2"]),
    }
    for branch, rows in data.groupby("branch", sort=False):
        for metric in METRICS:
            values = rows[metric]
            quantiles = values.quantile([0, .05, .25, .5, .75, .95, 1]).to_numpy()
            summaries.append({"branch": branch, "metric": metric, "count": len(values),
                              "mean": values.mean(), "sd": values.std(),
                              **dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"], quantiles))})
        for metric, (edges, labels) in bin_specs.items():
            counts = pd.cut(rows[metric], edges, labels=labels, include_lowest=True).value_counts(sort=False)
            if counts.sum() != len(rows):
                raise ValueError(f"Unbinned values: {metric}")
            bins.extend({"branch": branch, "metric": metric, "bin": label, "count": int(count),
                         "percent": count / len(rows) * 100} for label, count in counts.items())
        for video_id, group in rows.groupby("video_id"):
            videos.append({"branch": branch, "video_id": video_id, "retained_frames": len(group),
                           "confidence_median": group.confidence.median(),
                           "outside_nonzero_percent": (group.outside_area_percent > 0).mean() * 100,
                           "upsampled_percent": (group.maximum_scale > 1).mean() * 100,
                           "maximum_scale_median": group.maximum_scale.median()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "tables/retained_frame_quantiles.csv", index=False)
    pd.DataFrame(bins).to_csv(output / "tables/retained_frame_bins.csv", index=False)
    pd.DataFrame(videos).to_csv(output / "tables/retained_frame_per_video.csv", index=False)
    report = ["# Retained-frame distributions", "",
              f"Source: current saved audit of 100 videos, {total:,} source frames. No detection or decoding rerun.", "",
              "All distributions are frame-weighted; per-video statistics are exported separately.",
              "Confidence is an uncalibrated detector score, not a probability of complete face visibility.",
              "Outside area is the raw top-expanded bbox area outside the source / full expanded bbox area.",
              "ROI area is the actual crop rectangle area / source image area; it measures framing, not missing facial anatomy.",
              "Maximum scale = max(224 / crop width, 224 / crop height); >1 means at least one axis is enlarged.",
              "Alignment padding is the fraction of output pixels not fully supported by source pixels.", ""]
    for branch, rows in data.groupby("branch", sort=False):
        report += [f"## {branch}", "", f"Retained: {len(rows):,} ({len(rows) / total:.2%}); videos: {rows.video_id.nunique()}.",
                   f"Any raw-box clipping: {(rows.outside_area_percent > 0).mean():.2%}.",
                   f"Any-axis enlargement: {(rows.maximum_scale > 1).mean():.2%}; both-axis enlargement: {((rows.scale_x > 1) & (rows.scale_y > 1)).mean():.2%}.", "",
                   "| Metric | Mean | P5 | Median | P95 | Maximum |",
                   "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in summary.loc[summary.branch == branch].itertuples():
            report.append(f"| {row.metric} | {row.mean:.4f} | {row.p05:.4f} | {row.median:.4f} | {row.p95:.4f} | {row.max:.4f} |")
        report.append("")
    (output / "RETAINED_FRAME_QUALITY.md").write_text("\n".join(report) + "\n")
    return summary


def plot(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("confidence", "Detection confidence", np.linspace(.8, 1, 41)),
        ("outside_area_percent", "Expanded bbox outside image (%)", np.linspace(0, 10.00000001, 31)),
        ("maximum_scale", "Maximum-axis resize factor (x)", np.linspace(0, max(2, data.maximum_scale.max()), 41)),
        ("crop_width", "Crop width before resize (pixels)", np.linspace(0, data.crop_width.max() + 1, 41)),
        ("crop_height", "Crop height before resize (pixels)", np.linspace(0, data.crop_height.max() + 1, 41)),
        ("roi_area_percent", "Crop rectangle / source area (%)", np.linspace(0, data.roi_area_percent.max() + .01, 41)),
    ]
    for ax, (metric, label, edges) in zip(axes.flat, panels):
        for (branch, rows), color, name in zip(data.groupby("branch", sort=False), ["#267b9b", "#c96948"],
                                              ["Kalman + alignment", "No Kalman / alignment"]):
            counts, _ = np.histogram(rows[metric], edges)
            ax.stairs(counts / len(rows) * 100, edges, color=color, label=f"{name} (n={len(rows):,})", linewidth=1.8)
        ax.set(xlabel=label, ylabel="Retained frames per bin (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        if metric == "maximum_scale":
            ax.axvline(1, color="gray", ls="--", lw=1)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Retained-frame quality and resize distributions: 100-video audit", fontsize=15)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"figures/retained_frame_distributions.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/quality100")
    args = parser.parse_args()
    data, total = load_metrics(args.output_dir)
    summary = summarize(data, total, args.output_dir)
    plot(data, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Report: {args.output_dir / 'RETAINED_FRAME_QUALITY.md'}")


if __name__ == "__main__":
    main()
