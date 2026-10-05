"""Redraw quality previews using saved geometry; do not run detectors again."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .pipeline import RawRecording, VideoWriter
from .face_quality import clipped_crop_bounds
from .run_comparison import sha256
from .run_quality_audit import ROOT, comparison_preview, summarize


def render(metadata_path, output):
    cv2.setNumThreads(1)
    output = Path(output)
    metadata = json.loads(Path(metadata_path).read_text())
    for path, expected in metadata["source_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source changed: {path}")
    table = pd.read_csv(output / "tables/frames" / f"{metadata['video_id']}.csv")
    recording = RawRecording(metadata["raw_video_path"])
    if not np.array_equal(table.source_frame_index.to_numpy(), np.arange(len(recording.timestamps))):
        recording.close()
        raise ValueError("Source/table frame count mismatch")
    elapsed = recording.timestamps - recording.timestamps[0]
    if not np.allclose(table.source_elapsed_seconds, elapsed, atol=1e-8, rtol=0):
        recording.close()
        raise ValueError("Source/table elapsed timestamp mismatch")
    destination = output / "videos" / f"{metadata['video_id']}.mp4"
    temporary = destination.with_name(destination.stem + ".rendering.mp4")
    writer = VideoWriter(temporary, 896, 416, lossless=False)
    try:
        for index, row in enumerate(table.to_dict("records")):
            frame = recording.frame(index)
            aligned, unfiltered = (np.zeros((224, 224, 3), np.uint8) for _ in range(2))
            if frame is not None and row["unfiltered_direct_eligible"]:
                x1, y1, x2, y2 = clipped_crop_bounds(frame, [row[f"expanded_{key}"] for key in ("x1", "y1", "x2", "y2")])
                unfiltered = cv2.resize(frame[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_LINEAR)
            if frame is not None and row["aligned_eligible"]:
                matrix = np.array([[row[f"alignment_m{r}{c}"] for c in range(3)] for r in range(2)])
                aligned = cv2.warpAffine(frame, matrix, (224, 224), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            writer.write(comparison_preview(frame, aligned, unfiltered, row, index), float(elapsed[index]))
    finally:
        writer.close()
        recording.close()
    for path, expected in metadata["source_hashes"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source changed: {path}")
    temporary.replace(destination)
    print(f"[rendered] {metadata['video_id']} frames={len(table)}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/quality100")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    metadata_paths = sorted((args.output_dir / "tables/frames").glob("*.json"))
    if len(metadata_paths) != manifest["videos"]:
        raise ValueError("Metadata count differs from manifest")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(render, str(path), str(args.output_dir)) for path in metadata_paths]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"[progress] {len(results)}/{len(metadata_paths)}", flush=True)
    summarize(args.output_dir, sorted(results, key=lambda item: item["video_id"]),
              manifest["quality_gate"]["parameters"], manifest["selection_seed"])


if __name__ == "__main__":
    main()
