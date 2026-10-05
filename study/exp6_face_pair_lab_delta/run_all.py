"""Prepare Exp6 and dynamically schedule its nine tasks over available GPUs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex

from .build_dataset import prepare
from .config import (
    CACHE_DIR,
    FINETUNE_MAX_EPOCHS,
    HEAD_MAX_EPOCHS,
    OUTPUT_DIR,
    PATIENT_DIVERSE_30_40,
    SCHEDULE_ONLY_30_40,
    SEED,
    TARGETS,
    VIEWS,
    VIEWS_3,
)
from .plot_results import plot_results
from .train import train_task


_FRAME_INDEX = None
_GPU_ID = None
BATCH_ABLATION_VARIANT = "shared_patient_diverse_30_40"
SCHEDULE_ABLATION_VARIANT = "shared_schedule_30_40"
SCHEDULE_VARIANTS = (BATCH_ABLATION_VARIANT, SCHEDULE_ABLATION_VARIANT)


def _parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _worker_init(index_path, gpu_queue):
    global _FRAME_INDEX, _GPU_ID
    _GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(2)
    _FRAME_INDEX = FrameOffsetIndex.load(index_path)
    print(
        f"[worker-ready] pid={os.getpid()} gpu=cuda:{_GPU_ID} "
        f"indexed_frames={len(_FRAME_INDEX.starts)}", flush=True,
    )


def _worker_train(job):
    token = f"{job['seed']}:{job['target']}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    job_seed = (job["seed"] + offset) % (2**31 - 1)
    random.seed(job_seed)
    np.random.seed(job_seed)
    torch.manual_seed(job_seed)
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str})
    training_options = {
        "head_epochs": job["head_epochs"],
        "finetune_epochs": job["finetune_epochs"],
    }
    training_options.update(job.get("training_options", {}))
    metrics = train_task(
        target=job["target"], records=records, scaler=job["scaler"],
        frame_index=_FRAME_INDEX, device_id=_GPU_ID, run_dir=job["run_dir"],
        seed=job_seed, max_batches=job["max_batches"],
        model_variant=job["model_variant"],
        train_views=job["train_views"],
        **training_options,
    )
    return {"target": job["target"], "gpu": _GPU_ID, "metrics": metrics}


def _validate_records(records_by_target):
    patient_splits = {}
    for target, records in records_by_target.items():
        if records.groupby("hospital_id").split.nunique().max() != 1:
            raise AssertionError(f"Patient leakage for {target}")
        if not (records.second_lab_time_unix > records.first_lab_time_unix).all():
            raise AssertionError(f"Non-increasing lab times for {target}")
        if not (records.second_video_time_unix > records.first_video_time_unix).all():
            raise AssertionError(f"Non-increasing video times for {target}")
        if records.first_video_id.eq(records.second_video_id).any():
            raise AssertionError(f"Same-video pair for {target}")
        if not np.allclose(
            records.raw_delta,
            records.second_value - records.first_value,
            rtol=0.0, atol=1e-12,
        ):
            raise AssertionError(f"Incorrect delta labels for {target}")
        patient_splits[target] = records.groupby("hospital_id").split.first().to_dict()
    return patient_splits


def _smoke(records, scalers, frame_index, target, device, output_dir,
           model_variant, train_views, training_options):
    sample = records[target].groupby("split", group_keys=False).head(2).copy()
    # Keep all three split names while constraining each loader to a tiny sample.
    smoke_dir = output_dir / "smoke" / target
    shutil.rmtree(smoke_dir.parent, ignore_errors=True)
    smoke_options = {**training_options, "head_epochs": 1, "finetune_epochs": 1}
    train_task(
        target=target, records=sample, scaler=scalers[target],
        frame_index=frame_index, device_id=device, run_dir=smoke_dir,
        seed=SEED, max_batches=1,
        model_variant=model_variant, train_views=train_views,
        **smoke_options,
    )
    checkpoint = torch.load(smoke_dir / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("target") != target or not checkpoint.get("model_state_dict"):
        raise RuntimeError("Smoke checkpoint validation failed")
    shutil.rmtree(smoke_dir.parent)
    print(
        f"[smoke-ok] target={target} variant={model_variant} "
        "frozen+finetune+checkpoint", flush=True,
    )


def _load_prepared(targets):
    scaler_path = OUTPUT_DIR / "target_scalers.json"
    index_path = CACHE_DIR / "frame_offsets.npz"
    if not scaler_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            "The completed shared-backbone Exp6 data artifacts are unavailable"
        )
    scalers = json.loads(scaler_path.read_text(encoding="utf-8"))
    records = {}
    for target in targets:
        path = OUTPUT_DIR / "task_records" / f"{target}.csv"
        if not path.is_file() or target not in scalers:
            raise FileNotFoundError(f"Missing prepared Exp6 target: {target}")
        records[target] = pd.read_csv(path, dtype={"hospital_id": str})
    return records, {target: scalers[target] for target in targets}, FrameOffsetIndex.load(index_path)


def _validate_against_baseline(records_by_target):
    index = pd.read_csv(OUTPUT_DIR / "run_index.csv")
    if (set(index.target) != set(records_by_target)
            or len(index) != len(records_by_target)
            or not index.status.eq("ok").all()):
        raise RuntimeError("Exp6 baseline runs are incomplete")
    for target, records in records_by_target.items():
        reference = pd.read_csv(
            OUTPUT_DIR / "runs" / target / "pair_predictions.csv",
            dtype={"pair_id": str, "hospital_id": str},
        )
        columns = ("pair_id", "hospital_id", "split", "first_video_id", "second_video_id")
        expected = records.sort_values("pair_id").reset_index(drop=True)
        reference = reference.sort_values("pair_id").reset_index(drop=True)
        if len(expected) != len(reference) or expected.pair_id.duplicated().any():
            raise AssertionError(f"Baseline pair count differs for {target}")
        if any(not expected[column].astype(str).equals(reference[column].astype(str))
               for column in columns):
            raise AssertionError(f"Baseline pair identity or split differs for {target}")
        if not np.allclose(expected.raw_delta, reference.raw_delta, rtol=0, atol=1e-10):
            raise AssertionError(f"Baseline raw labels differ for {target}")


def _write_variant_manifest(output_dir, targets, records_by_target, variant,
                            train_views, training_options=None):
    source_manifest = OUTPUT_DIR / "experiment_manifest.json"
    task_fingerprints = {}
    for target in targets:
        path = OUTPUT_DIR / "task_records" / f"{target}.csv"
        task_fingerprints[target] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pairs": int(len(records_by_target[target])),
            "patients": int(records_by_target[target].hospital_id.nunique()),
        }
    manifest = {
        "schema_version": 1,
        "experiment": f"exp6_paired_face_lab_delta_{variant}",
        "model_variant": "shared" if variant in ("shared_views3", *SCHEDULE_VARIANTS) else variant,
        "training_views": list(train_views),
        "controlled_difference": (
            "training uses original, horizontal flip, and center crop only"
            if variant == "shared_views3" else
            "training batches mix up to 12 patients with two paired frame rows "
            "per patient; head/fine-tune learning rates, cosine floors, epoch "
            "limits, and patience match Exp2 patient_diverse_schedule_30_40"
            if variant == BATCH_ABLATION_VARIANT else
            "baseline chunk-shuffled training batches; only head/fine-tune "
            "learning rates, cosine floors, epoch limits, and patience match "
            "the patient-diverse 30/40 variant"
            if variant == SCHEDULE_ABLATION_VARIANT else
            "early and late faces use separately parameterized EfficientNet-B0 "
            "backbones initialized from the same ImageNet checkpoint"
        ),
        "unchanged": (
            ["task records", "patient split", "train-only target scaler",
             "20 selected frames", "synchronized five views", "difference fusion",
             "32-dimensional head", "AdamW and weight decay", "loss weighting"]
            if variant in SCHEDULE_VARIANTS else
            ["task records", "patient split", "train-only target scaler",
             "20 selected frames", "synchronized views", "difference fusion",
             "32-dimensional head", "optimizer", "learning rates", "early stopping"]
        ),
        "training_options": training_options or {},
        "source_experiment_manifest": {
            "path": str(source_manifest.resolve()),
            "sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        },
        "task_records": task_fingerprints,
        "target_scalers_sha256": hashlib.sha256(
            (OUTPUT_DIR / "target_scalers.json").read_bytes()
        ).hexdigest(),
        "frame_index_sha256": hashlib.sha256(
            (CACHE_DIR / "frame_offsets.npz").read_bytes()
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    """Run only the native-224 shared-backbone regression protocol."""
    from study.common.face_video import face_source_mode
    from study.common.rerun_face224 import experiment_plan, INDEX_DIR, run_experiment
    if face_source_mode() != "face224":
        raise ValueError("Exp6 regression no longer accepts legacy 128 videos")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, choices=(24, 12, 6), default=24)
    args = parser.parse_args()
    job = next(item for item in experiment_plan() if item["key"] == f"exp6_delta_{args.hours}h")
    run_experiment(job, FrameOffsetIndex.load(INDEX_DIR / "frame_offsets.npz"))


if __name__ == "__main__":
    main()
