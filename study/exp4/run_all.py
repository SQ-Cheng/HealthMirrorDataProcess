"""Prepare, validate, schedule, train, and summarize Exp4."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from .build_dataset import add_balanced_patient_split, build_recovery_candidates
from .config import CACHE_DIR, FRAMES_PER_VIDEO, OUTPUT_DIR, SEED, TRAIN_VIEWS
from .frame_index import FrameOffsetIndex, build_or_reuse_frame_index
from .models import build_model, freeze_backbone
from .plot_results import plot_results
from .train import _loader, _prepare_images, _weighted_loss, train_seed


def prepare(seed=SEED):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates, quality = build_recovery_candidates(OUTPUT_DIR)
    frame_index, frame_manifest = build_or_reuse_frame_index(candidates, CACHE_DIR / "frames20")
    usable = set(frame_index.video_lookup)
    records = candidates.loc[candidates.video_id.astype(str).isin(usable)].copy()
    excluded = candidates.loc[~candidates.video_id.astype(str).isin(usable), ["video_id", "video_path"]]
    excluded.to_csv(OUTPUT_DIR / "frame_exclusions.csv", index=False)
    if len(records) < 100 or records.hospital_id.nunique() < 50:
        raise RuntimeError(f"Insufficient usable recovery cohort: {len(records)} videos")
    records, split_manifest = add_balanced_patient_split(
        records, OUTPUT_DIR, seed=seed
    )
    if records.groupby("hospital_id").split.nunique().gt(1).any():
        raise AssertionError("Patient leakage after split")
    if not records.recovery_score.between(0, 1).all():
        raise AssertionError("Recovery labels outside [0,1]")
    quality["counts"].update({
        "frame_usable_videos": len(records),
        "frame_usable_patients": int(records.hospital_id.nunique()),
        "frame_excluded_videos": len(excluded),
    })
    quality["frame_policy"] = frame_manifest["policy"]
    quality["split_policy"] = split_manifest
    quality["training_input"] = {
        "frames_per_video": FRAMES_PER_VIDEO,
        "train_views": list(TRAIN_VIEWS),
        "evaluation_views": ["original"],
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records, frame_index


def smoke_test(records, frame_index, device_id=0):
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    train_records = records.loc[records.split.eq("train")].head(8).reset_index(drop=True)
    dataset, loader = _loader(frame_index, train_records, True)
    images, labels, _, view_codes, weights = next(iter(loader))
    images = _prepare_images(images, view_codes, device)
    repeat = view_codes.shape[1]
    labels = labels.repeat_interleave(repeat).to(device)
    weights = weights.repeat_interleave(repeat).to(device)
    model, _, _ = build_model(); freeze_backbone(model)
    model = model.to(device, memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predictions = model(images).squeeze(1)
        loss = _weighted_loss(predictions, labels, weights)
    loss.backward()
    if not torch.isfinite(loss) or predictions.min() < 0 or predictions.max() > 1:
        raise RuntimeError("Smoke test produced invalid outputs")
    print(
        f"[smoke-ok] source_frames={len(dataset)} model_inputs={len(images)} "
        f"shape={tuple(images.shape)} loss={float(loss):.6f} views={TRAIN_VIEWS}", flush=True,
    )


def worker(seed, device_id, run_dir):
    records = pd.read_csv(OUTPUT_DIR / "records.csv", dtype={"hospital_id": str})
    frame_index = FrameOffsetIndex.load(CACHE_DIR / "frames20/frame_offsets.npz")
    train_seed(records, frame_index, seed, device_id, run_dir)


def _clear_stale_training_outputs():
    runs_dir = OUTPUT_DIR / "runs"
    if runs_dir.is_dir():
        shutil.rmtree(runs_dir)
    stale_files = (
        "history.csv",
        "metrics.csv",
        "metrics_all.csv",
        "model.pt",
        "video_predictions.csv",
        "training_history.png",
        "run_manifest.json",
        "run_index.csv",
        "test_metrics_seed_summary.csv",
        "test_metrics_summary.csv",
        "interpretability_examples.csv",
        "interpretability_maps.npz",
        "interpretability_manifest.json",
    )
    for name in stale_files:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    for name in (
        "results_summary.png",
        "interpretability_gradcam.png",
        "interpretability_occlusion.png",
        "interpretability_gradcam_occlusion.png",
    ):
        (OUTPUT_DIR / "figures" / name).unlink(missing_ok=True)


def schedule(seed=SEED, device_id=0):
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Exp4 formal run requires one CUDA GPU")
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device {device_id}")
    _clear_stale_training_outputs()
    run_dir = OUTPUT_DIR
    command = [
        sys.executable, "-m", "study.exp4.run_all", "--worker",
        "--seed", str(seed), "--device", str(device_id), "--run-dir", str(run_dir),
    ]
    print(f"[launch] seed={seed} gpu={device_id} jobs=1", flush=True)
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2])
    code = process.wait()
    print(f"[worker-exit] seed={seed} gpu={device_id} code={code}", flush=True)
    if code:
        raise RuntimeError(f"Exp4 worker failed: seed={seed} gpu={device_id} code={code}")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics.to_csv(OUTPUT_DIR / "metrics_all.csv", index=False)
    pd.DataFrame([{
        "seed": seed,
        "device": f"cuda:{device_id}",
        "status": "ok",
        "run_dir": str(run_dir),
    }]).to_csv(OUTPUT_DIR / "run_index.csv", index=False)
    plot_results(OUTPUT_DIR)
    print("[experiment-complete] selected seed finished and figures generated", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.worker:
        worker(args.seed, args.device, args.run_dir)
        return
    records, frame_index = prepare(args.seed)
    if args.smoke:
        smoke_test(records, frame_index, args.device)
        return
    if not args.prepare_only:
        schedule(args.seed, args.device)


if __name__ == "__main__":
    main()
