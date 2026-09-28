"""Train a central ten-second clip with reduced two-stage learning rates."""

import json

import cv2
import pandas as pd

from .clips import build_or_reuse_index
from .config import OUTPUT_DIR
from .plot_ablations import plot_comparison
from .prepare import prepare
from .run_ablations import (ABLATION_DIR, _run_variant, _source_records)


NAME = "middle10s_low_lr_20_30"
INDEX_DIR_10S = OUTPUT_DIR.parent / "cache/middle10s_index"
FRAMES_PER_SECOND = 30
CLIP_FRAMES = 10 * FRAMES_PER_SECOND
SCHEDULE = {
    "head_epochs": 20, "head_patience": 4,
    "head_lr": 1e-4, "head_min_lr": 1e-5,
    "finetune_epochs": 30, "finetune_patience": 6,
    "finetune_lr": 1e-6, "finetune_min_lr": 1e-7,
}


def _verify_source_fps(index):
    for path in index.video_paths:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 30.0) > 1e-3:
                raise ValueError(f"Expected 30 fps source video: {path}")
        finally:
            capture.release()
    print(f"[fps-validated] videos={len(index.video_paths)} fps=30", flush=True)


def main():
    if not (OUTPUT_DIR / "COMPLETE").is_file():
        raise RuntimeError("Exp3 baseline is incomplete")
    for reference in ("patient_diverse_schedule_30_40", "middle48"):
        if not (ABLATION_DIR / reference / "COMPLETE").is_file():
            raise RuntimeError(f"Required comparison is incomplete: {reference}")
    if (ABLATION_DIR / NAME).exists():
        raise FileExistsError(f"Ablation output exists: {ABLATION_DIR / NAME}")
    baseline_index = prepare()
    _verify_source_fps(baseline_index)
    source = _source_records()
    videos = pd.concat(
        [frame[["video_id", "mirror", "lab_patient_id"]]
         for frame in source.values()], ignore_index=True,
    )
    index = build_or_reuse_index(
        videos, INDEX_DIR_10S, clip_frames=CLIP_FRAMES, positions=(0.5,)
    )
    print(f"[middle10s-index] videos={len(index.video_ids)} "
          f"clips={len(index.starts)} clip_frames={index.starts.shape[1]}",
          flush=True)
    with open(OUTPUT_DIR / "target_scalers.json", encoding="utf-8") as handle:
        scalers = json.load(handle)["targets"]
    _run_variant(NAME, index, INDEX_DIR_10S / "clip_offsets.npz",
                 source, scalers, "random_video", SCHEDULE,
                 positions=(0.5,))
    plot_comparison(
        OUTPUT_DIR,
        (ABLATION_DIR / "patient_diverse_schedule_30_40",
         ABLATION_DIR / "middle48", ABLATION_DIR / NAME),
        labels=("Baseline 16f", "Diverse + 30/40", "Middle 48f", "Middle 10s"),
        output_name="comparison_including_10s",
    )
    print("[middle10s-ablation-complete]", flush=True)


if __name__ == "__main__":
    main()
