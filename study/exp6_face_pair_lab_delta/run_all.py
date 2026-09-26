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


def _smoke(records, scalers, frame_index, target, device, output_dir, model_variant, train_views):
    sample = records[target].groupby("split", group_keys=False).head(2).copy()
    # Keep all three split names while constraining each loader to a tiny sample.
    smoke_dir = output_dir / "smoke" / target
    shutil.rmtree(smoke_dir.parent, ignore_errors=True)
    train_task(
        target=target, records=sample, scaler=scalers[target],
        frame_index=frame_index, device_id=device, run_dir=smoke_dir,
        seed=SEED, head_epochs=1, finetune_epochs=1, max_batches=1,
        model_variant=model_variant, train_views=train_views,
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
        "model_variant": "shared" if variant in ("shared_views3", BATCH_ABLATION_VARIANT) else variant,
        "training_views": list(train_views),
        "controlled_difference": (
            "training uses original, horizontal flip, and center crop only"
            if variant == "shared_views3" else
            "training batches mix up to 12 patients with two paired frame rows "
            "per patient; head/fine-tune learning rates, cosine floors, epoch "
            "limits, and patience match Exp2 patient_diverse_schedule_30_40"
            if variant == BATCH_ABLATION_VARIANT else
            "early and late faces use separately parameterized EfficientNet-B0 "
            "backbones initialized from the same ImageNet checkpoint"
        ),
        "unchanged": (
            ["task records", "patient split", "train-only target scaler",
             "20 selected frames", "synchronized five views", "difference fusion",
             "32-dimensional head", "AdamW and weight decay", "loss weighting"]
            if variant == BATCH_ABLATION_VARIANT else
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=HEAD_MAX_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_MAX_EPOCHS)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--variant", choices=("shared", "independent_backbones", "shared_views3",
                              BATCH_ABLATION_VARIANT),
        default="shared"
    )
    args = parser.parse_args()
    targets = _parse_csv(args.targets)
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    if args.variant == BATCH_ABLATION_VARIANT and set(targets) != set(TARGETS):
        raise ValueError("The patient-diverse comparison requires all nine baseline targets")

    if args.variant == "shared":
        records_by_target, scalers, frame_index = prepare(targets)
        output_dir = OUTPUT_DIR
    else:
        records_by_target, scalers, frame_index = _load_prepared(targets)
        output_dir = OUTPUT_DIR / args.variant
        if args.variant == BATCH_ABLATION_VARIANT:
            _validate_against_baseline(records_by_target)
            if output_dir.exists() and not args.overwrite:
                raise FileExistsError(f"Ablation output already exists: {output_dir}")
        train_views = VIEWS_3 if args.variant == "shared_views3" else VIEWS
        training_options = (
            PATIENT_DIVERSE_30_40 if args.variant == BATCH_ABLATION_VARIANT else {}
        )
        _write_variant_manifest(
            output_dir, targets, records_by_target, args.variant, train_views,
            training_options,
        )
    if args.variant == "shared":
        train_views = VIEWS
        training_options = {}
    model_variant = (
        "shared" if args.variant in ("shared_views3", BATCH_ABLATION_VARIANT)
        else args.variant
    )
    _validate_records(records_by_target)
    if args.prepare_only:
        return
    if args.smoke:
        _smoke(
            records_by_target, scalers, frame_index, targets[0], args.device,
            output_dir, model_variant, train_views,
        )
        return

    runs_dir = output_dir / "runs"
    if args.overwrite:
        shutil.rmtree(runs_dir, ignore_errors=True)
        for name in ("metrics_all.csv", "run_index.csv"):
            (output_dir / name).unlink(missing_ok=True)
        shutil.rmtree(output_dir / "figures", ignore_errors=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    jobs = [{
        "target": target,
        "records_path": str(OUTPUT_DIR / "task_records" / f"{target}.csv"),
        "scaler": scalers[target],
        "run_dir": str(runs_dir / target),
        "seed": SEED,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "max_batches": args.max_batches,
        "model_variant": model_variant,
        "train_views": train_views,
        "training_options": training_options,
    } for target in targets]
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Exp6 requires CUDA")
    worker_count = min(args.workers or gpu_count, gpu_count, len(jobs))
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in range(worker_count):
        gpu_queue.put(gpu_id)
    print(
        f"[scheduler] variant={args.variant} jobs={len(jobs)} workers={worker_count} "
        f"gpus={gpu_count} dynamic_queue=true", flush=True,
    )
    run_rows, metric_frames, failures = [], [], []
    index_path = str(CACHE_DIR / "frame_offsets.npz")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_worker_init,
        initargs=(index_path, gpu_queue),
    ) as executor:
        futures = {executor.submit(_worker_train, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                metric_frames.append(pd.DataFrame(result["metrics"]))
                run_rows.append({
                    "target": result["target"], "status": "ok",
                    "gpu": result["gpu"], "run_dir": job["run_dir"],
                })
                print(
                    f"[scheduler-complete] target={result['target']} "
                    f"gpu=cuda:{result['gpu']}", flush=True,
                )
            except Exception as exc:
                failure = {
                    "target": job["target"], "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                run_rows.append(failure)
                print(
                    f"[scheduler-failed] target={job['target']} "
                    f"error={failure['error']}", flush=True,
                )
    pd.DataFrame(run_rows).to_csv(output_dir / "run_index.csv", index=False)
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(
            output_dir / "metrics_all.csv", index=False
        )
    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} Exp6 jobs failed")
    (output_dir / "failures.json").unlink(missing_ok=True)
    plot_results(output_dir)
    if args.variant == BATCH_ABLATION_VARIANT:
        from .plot_patient_diverse_comparison import plot_comparison
        plot_comparison(OUTPUT_DIR, output_dir, targets)
        (output_dir / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(
        f"[experiment-complete] variant={args.variant} "
        "all nine tasks and figures generated", flush=True,
    )


if __name__ == "__main__":
    main()
