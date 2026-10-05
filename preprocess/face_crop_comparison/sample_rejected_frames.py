"""Render a reproducible random sample of rejected frames from saved geometry."""

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .analyze_retained_frames import retained_mask
from .face_quality import clipped_crop_bounds
from .pipeline import RawRecording
from .run_quality_audit import ROOT, comparison_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/quality100")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("Sample size must be positive")
    cv2.setNumThreads(1)
    selected = pd.read_csv(args.output_dir / "tables/selected_videos.csv", dtype={"hospital_id": str})
    rejected = []
    for video_id in selected.video_id:
        table = pd.read_csv(args.output_dir / f"tables/frames/{video_id}.csv", dtype={"hospital_id": str})
        rejected.append(table.loc[~retained_mask(table.aligned_eligible)])
    population = pd.concat(rejected, ignore_index=True)
    sample = population.sample(n=args.count, replace=False, random_state=args.seed).reset_index(drop=True)
    sample["sample_id"] = np.arange(1, len(sample) + 1)
    sample["image_path"] = [f"figures/rejected_random{args.count}/{i:03d}_{row.video_id}_{int(row.source_frame_index):05d}.jpg"
                            for i, row in enumerate(sample.itertuples(), 1)]
    image_dir = args.output_dir / f"figures/rejected_random{args.count}"
    image_dir.mkdir(parents=True, exist_ok=True)
    # Only clean this dedicated sample folder, never existing audit outputs.
    for path in image_dir.glob("*.jpg"):
        path.unlink()
    sources = selected.set_index("video_id")
    for video_id, group in sample.groupby("video_id", sort=False):
        recording = RawRecording(sources.loc[video_id, "raw_video_path"])
        try:
            for row in group.to_dict("records"):
                index = int(row["source_frame_index"])
                elapsed = float(recording.timestamps[index] - recording.timestamps[0])
                if abs(elapsed - row["source_elapsed_seconds"]) > 1e-8:
                    raise ValueError(f"Source timestamp mismatch: {video_id}, {index}")
                frame = recording.frame(index)
                aligned, unfiltered = (np.zeros((224, 224, 3), np.uint8) for _ in range(2))
                if frame is None and row["rejection_reason"] != "source_decode_failed":
                    raise ValueError(f"Unexpected source decode failure: {video_id}, {index}")
                if row["aligned_eligible"]:
                    raise ValueError("Sample contains an accepted aligned frame")
                if frame is not None and row["unfiltered_direct_eligible"]:
                    box = [row[f"expanded_{key}"] for key in ("x1", "y1", "x2", "y2")]
                    x1, y1, x2, y2 = clipped_crop_bounds(frame, box)
                    unfiltered = cv2.resize(frame[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_LINEAR)
                preview = comparison_preview(frame, aligned, unfiltered, row, index)
                destination = args.output_dir / row["image_path"]
                if not cv2.imwrite(str(destination), preview, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Image write failed: {destination}")
                if cv2.imread(str(destination)).shape != (416, 896, 3):
                    raise ValueError(f"Preview dimensions differ: {destination}")
        finally:
            recording.close()
    columns = ["sample_id", "video_id", "hospital_id", "source_frame_index", "source_elapsed_seconds",
               "confidence", "top_detection_confidence", "expanded_box_outside_area_fraction",
               "rejection_reason", "aligned_eligible", "unfiltered_direct_eligible", "image_path"]
    sample.reindex(columns=columns).to_csv(args.output_dir / f"tables/rejected_random{args.count}.csv", index=False)
    counts = sample.rejection_reason.value_counts().rename_axis("reason").reset_index(name="sample_frames")
    counts.to_csv(args.output_dir / f"tables/rejected_random{args.count}_reasons.csv", index=False)
    cards = []
    for row in sample.itertuples():
        label = f"{row.sample_id:03d} | {row.video_id} | frame {int(row.source_frame_index)} | {row.rejection_reason}"
        cards.append(f'<figure><figcaption>{html.escape(label)}</figcaption>'
                     f'<a href="{html.escape(row.image_path)}"><img loading="lazy" '
                     f'src="{html.escape(row.image_path)}" width="896" height="416"></a></figure>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Random rejected frames</title><style>'
            'body{font-family:Arial,sans-serif;margin:24px;background:#fafafa;color:#222}'
            'figure{margin:20px 0}img{max-width:100%;height:auto}figcaption{margin-bottom:6px}'
            '</style><h1>Random rejected frames</h1>'
            f'<p>{args.count} frames sampled without replacement from {len(population):,} aligned-branch rejections; seed {args.seed}.</p>'
            '<p>Original + boxes | Kalman + alignment | No Kalman / alignment. Rejected crops remain black, matching the video previews.</p>'
            + "\n".join(cards) + '</html>')
    gallery = args.output_dir / f"rejected_random{args.count}.html"
    gallery.write_text(page)
    (args.output_dir / f"tables/rejected_random{args.count}_manifest.json").write_text(json.dumps({
        "seed": args.seed, "sample_size": args.count, "population_size": len(population),
        "population_rule": "aligned_eligible == false", "sampling": "uniform frames without replacement",
        "videos_in_sample": sample.video_id.nunique(), "layout": "same 896x416 three-column video preview",
        "detector_rerun": False,
    }, indent=2) + "\n")
    print(counts.to_string(index=False))
    print(f"Images: {args.count}; videos: {sample.video_id.nunique()}; gallery: {gallery}")


if __name__ == "__main__":
    main()
