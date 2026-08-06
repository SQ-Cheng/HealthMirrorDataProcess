"""Prepare, validate, schedule, train, and summarize Exp4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from .build_dataset import add_balanced_patient_split, build_recovery_candidates
from .config import CACHE_DIR, OUTPUT_DIR, SEEDS, TRAIN_VIEWS
from .frame_index import FrameOffsetIndex, build_or_reuse_frame_index
from .models import build_model, freeze_backbone
from .plot_results import plot_results
from .train import _loader, _prepare_images, _weighted_loss, train_seed


def prepare():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates, quality = build_recovery_candidates(OUTPUT_DIR)
    frame_index, frame_manifest = build_or_reuse_frame_index(candidates, CACHE_DIR / "frames20")
    usable = set(frame_index.video_lookup)
    records = candidates.loc[candidates.video_id.astype(str).isin(usable)].copy()
    excluded = candidates.loc[~candidates.video_id.astype(str).isin(usable), ["video_id", "video_path"]]
    excluded.to_csv(OUTPUT_DIR / "frame_exclusions.csv", index=False)
    if len(records) < 100 or records.hospital_id.nunique() < 50:
        raise RuntimeError(f"Insufficient usable recovery cohort: {len(records)} videos")
    records, split_manifest = add_balanced_patient_split(records, OUTPUT_DIR)
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


def schedule():
    if torch.cuda.device_count() < 4:
        raise RuntimeError(f"Exp4 formal run requires four GPUs, found {torch.cuda.device_count()}")
    processes = []
    for device_id, seed in enumerate(SEEDS):
        run_dir = OUTPUT_DIR / "runs" / f"seed_{seed}"
        command = [
            sys.executable, "-m", "study.exp4.run_all", "--worker",
            "--seed", str(seed), "--device", str(device_id), "--run-dir", str(run_dir),
        ]
        print(f"[launch] seed={seed} gpu={device_id}", flush=True)
        processes.append((seed, device_id, subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2])))
    failures = []
    for seed, device_id, process in processes:
        code = process.wait()
        print(f"[worker-exit] seed={seed} gpu={device_id} code={code}", flush=True)
        if code:
            failures.append((seed, device_id, code))
    if failures:
        raise RuntimeError(f"Exp4 workers failed: {failures}")
    metrics = []
    run_rows = []
    for device_id, seed in enumerate(SEEDS):
        run_dir = OUTPUT_DIR / "runs" / f"seed_{seed}"
        metrics.append(pd.read_csv(run_dir / "metrics.csv"))
        run_rows.append({"seed": seed, "device": f"cuda:{device_id}", "status": "ok", "run_dir": str(run_dir)})
    pd.concat(metrics, ignore_index=True).to_csv(OUTPUT_DIR / "metrics_all.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "run_index.csv", index=False)
    plot_results(OUTPUT_DIR)
    print("[experiment-complete] all seeds finished and figures generated", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.worker:
        worker(args.seed, args.device, args.run_dir)
        return
    records, frame_index = prepare()
    if args.smoke:
        smoke_test(records, frame_index, args.device)
        return
    if not args.prepare_only:
        schedule()


if __name__ == "__main__":
    main()
