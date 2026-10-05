"""Rebuild and train Exp6 pairs with shorter video/lab matching windows."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex

from .build_dataset import _add_split_and_scaling, _pair_target
from .config import (
    CACHE_DIR, FINETUNE_MAX_EPOCHS, HEAD_MAX_EPOCHS, OUTPUT_DIR, SEED,
    TARGETS, VIEWS, WEIGHTS_DIR, PRETRAINED_WEIGHT_FILE,
)
from .plot_match_window_comparison import plot_comparison
from .plot_results import plot_results
from .run_all import _validate_records, _worker_init, _worker_train


ABLATIONS = OUTPUT_DIR / "ablations"
SOURCE = OUTPUT_DIR / "source_data/base_manifest.csv"
INDEX = CACHE_DIR / "frame_offsets.npz"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output(hours):
    return ABLATIONS / f"lab_match_{hours}h"


def prepare_window(hours):
    output = _output(hours)
    if output.exists():
        raise FileExistsError(output)
    baseline = pd.read_csv(OUTPUT_DIR / "run_index.csv")
    if (len(baseline) != len(TARGETS) or set(baseline.target) != set(TARGETS)
            or not baseline.status.eq("ok").all()):
        raise RuntimeError("The nine-task 24h Exp6 baseline is incomplete")
    base = pd.read_csv(SOURCE, dtype={"hospital_id": str, "video_id": str})
    index = FrameOffsetIndex.load(INDEX)
    usable = set(index.video_lookup)
    records_by_target, scalers, selections, summaries, split_rows = {}, {}, {}, [], []
    for target in TARGETS:
        original, _ = _pair_target(base, target)
        original = original.loc[
            original.first_video_id.isin(usable) & original.second_video_id.isin(usable)
        ]
        saved = pd.read_csv(OUTPUT_DIR / "task_records" / f"{target}.csv")
        if set(original.pair_id) != set(saved.pair_id):
            raise AssertionError(f"The saved 24h pair source has changed: {target}")

        pairs, summary = _pair_target(base, target, max_match_delta_hours=hours)
        valid = pairs.first_video_id.isin(usable) & pairs.second_video_id.isin(usable)
        summary["pairs_excluded_invalid_frames"] = int((~valid).sum())
        pairs = pairs.loc[valid].reset_index(drop=True)
        if (pairs.first_match_delta_h.gt(hours + 1e-9).any()
                or pairs.second_match_delta_h.gt(hours + 1e-9).any()):
            raise AssertionError(f"Out-of-window laboratory match: {target}")
        records, scaler, audit, selection = _add_split_and_scaling(pairs, target, SEED)
        records_by_target[target] = records
        scalers[target] = scaler
        selections[target] = selection
        split_rows.extend(audit)
        summary.update({
            "usable_pairs": len(records),
            "usable_patients": records.hospital_id.nunique(),
            "usable_videos": len(set(records.first_video_id) | set(records.second_video_id)),
        })
        summaries.append(summary)
        print(f"[prepared] hours={hours} target={target} pairs={len(records)} "
              f"patients={records.hospital_id.nunique()} split_seed_candidate="
              f"{selection['selected_candidate']}", flush=True)
    _validate_records(records_by_target)
    all_videos = set()
    for records in records_by_target.values():
        all_videos.update(records.first_video_id)
        all_videos.update(records.second_video_id)
    if any(index.frame_range(video_id)[1] - index.frame_range(video_id)[0] != 20
           for video_id in all_videos):
        raise AssertionError("A selected video lacks 20 indexed frames")

    task_dir = output / "task_records"
    task_dir.mkdir(parents=True)
    for target, records in records_by_target.items():
        records.to_csv(task_dir / f"{target}.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "task_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_distribution.csv", index=False)
    (output / "target_scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf-8")
    (output / "experiment_manifest.json").write_text(json.dumps({
        "schema_version": 1, "experiment": f"exp6_shared_lab_match_{hours}h",
        "matching_window_hours": hours,
        "source_base_manifest": str(SOURCE.resolve()),
        "source_sha256": _sha256(SOURCE),
        "frame_index": str(INDEX.resolve()),
        "frame_index_sha256": _sha256(INDEX),
        "baseline_output": str(OUTPUT_DIR.resolve()),
        "pair_policy": "filter each video/lab match before choosing one video per event and pairing consecutive eligible events",
        "split_policy": "independent patient-disjoint 512-candidate delta-distribution search",
        "split_selections": selections,
        "scaling": "train-only median/IQR of raw laboratory delta",
        "unchanged": ["source snapshot", "nine targets", "shared EfficientNet-B0 encoder",
                      "20 indexed frames per video", "five synchronized views",
                      "two-stage optimization", "job seeds", "loss weighting"],
    }, indent=2), encoding="utf-8")
    (output / "PREPARED").write_text("ok\n", encoding="ascii")
    print(f"[window-prepared] hours={hours} output={output}", flush=True)


def _load_prepared(hours):
    output = _output(hours)
    if not (output / "PREPARED").is_file() or (output / "COMPLETE").exists():
        raise RuntimeError(f"Window data are absent or already complete: {output}")
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    if (manifest["matching_window_hours"] != hours
            or manifest["source_sha256"] != _sha256(SOURCE)
            or manifest["frame_index_sha256"] != _sha256(INDEX)):
        raise AssertionError("Prepared source or frame index changed")
    scalers = json.loads((output / "target_scalers.json").read_text())
    records = {}
    for target in TARGETS:
        frame = pd.read_csv(output / "task_records" / f"{target}.csv",
                            dtype={"hospital_id": str})
        if (frame.first_match_delta_h.gt(hours + 1e-9).any()
                or frame.second_match_delta_h.gt(hours + 1e-9).any()
                or set(frame.split) != {"train", "val", "test"}):
            raise AssertionError(f"Invalid prepared window records: {target}")
        scaler = scalers[target]
        if not np.allclose(frame.scaled_delta,
                           (frame.raw_delta - scaler["median"]) / scaler["iqr"],
                           rtol=0, atol=1e-8):
            raise AssertionError(f"Incorrect saved target scaler: {target}")
        records[target] = frame
    _validate_records(records)
    if not (WEIGHTS_DIR / PRETRAINED_WEIGHT_FILE).is_file():
        raise FileNotFoundError(WEIGHTS_DIR / PRETRAINED_WEIGHT_FILE)
    return output, records, scalers


def train_window(hours):
    output, records, scalers = _load_prepared(hours)
    if (output / "runs").exists() or (output / "run_index.csv").exists():
        raise FileExistsError(f"Training artifacts already exist: {output}")
    gpu_count = min(torch.cuda.device_count(), 4, len(TARGETS))
    if gpu_count < 1:
        raise RuntimeError("Exp6 requires CUDA")
    (output / "runs").mkdir()
    jobs = [{
        "target": target,
        "records_path": str(output / "task_records" / f"{target}.csv"),
        "scaler": scalers[target],
        "run_dir": str(output / "runs" / target),
        "seed": SEED,
        "head_epochs": HEAD_MAX_EPOCHS,
        "finetune_epochs": FINETUNE_MAX_EPOCHS,
        "max_batches": None,
        "model_variant": "shared",
        "train_views": VIEWS,
        "training_options": {},
    } for target in TARGETS]
    context = mp.get_context("spawn")
    manager = context.Manager()
    queue = manager.Queue()
    for gpu in range(gpu_count):
        queue.put(gpu)
    rows, metrics, failures = [], [], []
    print(f"[scheduler-start] hours={hours} jobs={len(jobs)} gpus={gpu_count}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=gpu_count, mp_context=context,
                                 initializer=_worker_init, initargs=(str(INDEX), queue)) as executor:
            futures = {executor.submit(_worker_train, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    rows.append({"target": job["target"], "status": "ok",
                                 "gpu": result["gpu"], "run_dir": job["run_dir"]})
                    metrics.append(pd.DataFrame(result["metrics"]))
                except Exception as exc:
                    failure = {"target": job["target"], "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()}
                    failures.append(failure)
                    rows.append(failure)
                pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
                if metrics:
                    pd.concat(metrics, ignore_index=True).to_csv(
                        output / "metrics_all.csv", index=False)
                print(f"[scheduler] hours={hours} target={job['target']} "
                      f"status={rows[-1]['status']}", flush=True)
    finally:
        manager.shutdown()
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} Exp6 {hours}h tasks failed")
    plot_results(output)
    plot_comparison(OUTPUT_DIR, output, TARGETS, hours)
    (output / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[window-complete] hours={hours} output={output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, choices=(6, 12), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_window(args.hours)
    else:
        train_window(args.hours)


if __name__ == "__main__":
    main()
