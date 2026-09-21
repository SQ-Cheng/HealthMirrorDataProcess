"""Prepare, validate, train, and summarize Exp5."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from study.exp4.frame_index import FrameOffsetIndex, build_or_reuse_frame_index

from .build_dataset import (
    PROTOCOLS, finalize_protocol_records, prepare_candidates,
    protocol_output_dir,
)
from .config import ANALYTES, CACHE_DIR, OUTPUT_DIR, SEED, TARGET_COLUMN
from .data import PairedFrameDataset
from .models import build_model, freeze_backbone
from .plot_results import plot_results
from .train import _loss, _prepare, train


TRAINING_FILES = (
    "history.csv", "metrics.csv", "model.pt", "video_predictions.csv",
    "training_history.png", "run_manifest.json",
)


def prepare():
    candidates, frame_records, labs, shared = prepare_candidates(OUTPUT_DIR)
    frame_index, frame_manifest = build_or_reuse_frame_index(frame_records, CACHE_DIR)
    records_by_protocol = {}
    for protocol in PROTOCOLS:
        records, manifest = finalize_protocol_records(
            protocol, candidates[protocol], labs, frame_index.video_lookup,
            OUTPUT_DIR, shared,
        )
        if records.groupby("hospital_id").split.nunique().max() != 1:
            raise AssertionError(f"Patient leakage in {protocol}")
        components = [column for column in records if column.endswith("_deviation_component")]
        if len(components) != len(ANALYTES):
            raise AssertionError(f"Missing deviation components in {protocol}")
        if not np.allclose(records[components].mean(axis=1), records[TARGET_COLUMN]):
            raise AssertionError(f"Invalid equal-weight deviation score in {protocol}")
        manifest["frame_index"] = frame_manifest["policy"]
        run_dir = protocol_output_dir(OUTPUT_DIR, protocol)
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records_by_protocol[protocol] = records
    return records_by_protocol, frame_index


def smoke_test(records, frame_index, device_id):
    torch.cuda.set_device(device_id); device = torch.device(f"cuda:{device_id}")
    sample = records[records.split.eq("train")].head(4).reset_index(drop=True)
    dataset = PairedFrameDataset(frame_index, sample)
    pre, post, target, _, codes, weights = zip(*(dataset[index] for index in range(4)))
    pre = _prepare(torch.stack(pre), torch.stack(codes), device)
    post = _prepare(torch.stack(post), torch.stack(codes), device)
    target = torch.stack(target).to(device); weights = torch.stack(weights).to(device)
    model, _ = build_model(); freeze_backbone(model); model = model.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(pre, post).squeeze(1); loss = _loss(prediction, target, weights)
    loss.backward()
    if not torch.isfinite(loss) or not prediction.detach().ge(0).all():
        raise RuntimeError("Smoke test produced invalid output")
    print(f"[smoke-ok] pairs={len(pre)} shape={tuple(pre.shape)} loss={float(loss):.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    records = pd.read_csv(OUTPUT_DIR / "records.csv", dtype={"hospital_id": str})
    frame_index = FrameOffsetIndex.load(CACHE_DIR / "frame_offsets.npz")
    if args.smoke:
        smoke_test(records, frame_index, args.device); return
    for name in TRAINING_FILES:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    train(records, frame_index, args.seed, args.device, OUTPUT_DIR)
    plot_results(OUTPUT_DIR)
    print("[experiment-complete] checkpoint, histories, predictions, and figures generated", flush=True)


if __name__ == "__main__":
    main()
