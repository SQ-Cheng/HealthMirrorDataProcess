"""Prepare synchronized windows and dynamically schedule one Mamba per task."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import traceback

import mamba_ssm
import numpy as np
import pandas as pd
import torch

from .config import (
    D_CONV,
    D_MODEL,
    D_STATE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    ECG_MAX_INTERPOLATION_GAP_SECONDS,
    ECG_SAMPLE_RATE_HZ,
    ECG_SAMPLES_PER_WINDOW,
    ECG_TOTAL_STRIDE,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EXPAND,
    HEAD_HIDDEN_FEATURES,
    LEARNING_RATE,
    MAMBA_LAYERS,
    MAMBA_SSM_VERSION,
    MAX_EPOCHS,
    OUTPUT_DIR,
    RAW_DATA_ROOT,
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    SOURCE_DATA_DIR,
    TARGETS,
    TRAIN_BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WEIGHT_DECAY,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from .data import prepare_experiment_data
from .models import dependency_versions
from .train import train_task


_WORKER_GPU_ID = None


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _worker_init(gpu_queue):
    global _WORKER_GPU_ID
    _WORKER_GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_WORKER_GPU_ID)
    torch.set_num_threads(1)
    print(
        f"[worker-ready] pid={os.getpid()} gpu=cuda:{_WORKER_GPU_ID}",
        flush=True,
    )


def _worker_train(job):
    token = f"{job['seed']}:{job['target']}:video-ecg-mamba".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    job_seed = (int(job["seed"]) + offset) % (2**31 - 1)
    _set_seed(job_seed)
    metrics = train_task(
        target=job["target"],
        windows_path=job["windows_path"],
        run_dir=job["run_dir"],
        device=torch.device("cuda", _WORKER_GPU_ID),
        seed=job_seed,
        max_epochs=job["max_epochs"],
        patience_limit=job["patience"],
        max_batches=job["max_batches"],
    )
    return {
        "target": job["target"],
        "job_seed": job_seed,
        "run_dir": job["run_dir"],
        "metrics": metrics,
    }


def _collect_histories(output_dir):
    frames = []
    for target in TARGETS:
        path = os.path.join(output_dir, "runs", target, "history.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validate_mamba_runtime():
    if mamba_ssm.__version__ != MAMBA_SSM_VERSION:
        raise RuntimeError(
            f"Expected mamba-ssm {MAMBA_SSM_VERSION}, "
            f"found {mamba_ssm.__version__}; run install_mamba.sh"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Official mamba-ssm CUDA kernels require an NVIDIA GPU")
    test_model = mamba_ssm.Mamba(
        d_model=32, d_state=16, d_conv=4, expand=2
    ).cuda()
    test_input = torch.randn(1, 32, 32, device="cuda", requires_grad=True)
    test_model(test_input).mean().backward()
    if not torch.isfinite(test_input.grad).all():
        raise RuntimeError("mamba-ssm CUDA backward validation failed")
    del test_model, test_input
    torch.cuda.empty_cache()


def _parse_targets(values):
    if not values:
        return list(TARGETS)
    requested = []
    for value in values:
        requested.extend(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(requested) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}; valid targets={TARGETS}")
    return list(dict.fromkeys(requested))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=SOURCE_DATA_DIR)
    parser.add_argument("--raw-root", default=RAW_DATA_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected_targets = _parse_targets(args.targets)
    if args.smoke_test:
        args.max_epochs = 1
        args.patience = 1
        args.max_batches = args.max_batches or 2
    run_index_path = os.path.join(args.output_dir, "run_index.csv")
    if os.path.exists(run_index_path) and not args.overwrite:
        raise FileExistsError(
            f"Output already contains a run: {args.output_dir}. Use --overwrite."
        )
    if args.overwrite and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)

    _set_seed(args.seed)
    _validate_mamba_runtime()
    task_records, _, task_summary, recordings, quality = prepare_experiment_data(
        source_dir=args.source_dir,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        targets=selected_targets,
        seed=args.seed,
    )
    ready_targets = [target for target in selected_targets if target in task_records]
    if not ready_targets:
        raise RuntimeError("No requested task has valid synchronized data")
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 2,
                "experiment": (
                    "native_video_uniform_ecg_mamba_abnormal_score_regression"
                ),
                "task_type": "abnormal_score_regression",
                "targets": ready_targets,
                "seed": int(args.seed),
                "model": {
                    "mamba_implementation": "state-spaces/mamba",
                    "mamba_ssm_version": MAMBA_SSM_VERSION,
                    "d_model": D_MODEL,
                    "d_state": D_STATE,
                    "d_conv": D_CONV,
                    "expand": EXPAND,
                    "layers": MAMBA_LAYERS,
                    "head_hidden_features": HEAD_HIDDEN_FEATURES,
                    "dropout": DROPOUT,
                },
                "inputs": {
                    "window_seconds": WINDOW_SECONDS,
                    "window_stride_seconds": WINDOW_STRIDE_SECONDS,
                    "video_shape": [3, VIDEO_HEIGHT, VIDEO_WIDTH],
                    "video_interpolation": False,
                    "video_frame_sampling": False,
                    "ecg_resampling": {
                        "method": "linear interpolation on source Unix timestamps",
                        "sample_rate_hz": ECG_SAMPLE_RATE_HZ,
                        "samples_per_window": ECG_SAMPLES_PER_WINDOW,
                        "max_source_gap_seconds": (
                            ECG_MAX_INTERPOLATION_GAP_SECONDS
                        ),
                        "extrapolation": False,
                    },
                    "ecg_input_channels": ["robust_normalized_amplitude"],
                    "ecg_learned_token_stride": ECG_TOTAL_STRIDE,
                    "fusion_order": "actual_timestamp",
                },
                "training": {
                    "objective": {
                        "name": "SmoothL1Loss",
                        "beta": SMOOTH_L1_BETA,
                        "auxiliary_classification_loss": False,
                    },
                    "score_transform": SCORE_TRANSFORM,
                    "checkpoint_selection": (
                        "minimum video-level validation MAE"
                    ),
                    "batch_size": TRAIN_BATCH_SIZE,
                    "eval_batch_size": EVAL_BATCH_SIZE,
                    "train_data_workers_per_job": TRAIN_NUM_WORKERS,
                    "eval_data_workers_per_job": EVAL_NUM_WORKERS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "max_epochs": int(args.max_epochs),
                    "early_stopping_patience": int(args.patience),
                    "amp": "FP32 parameters with CUDA FP16 autocast",
                    "one_model_per_target": True,
                },
                "dependencies": dependency_versions(),
                "valid_recordings": int(len(recordings)),
                "excluded_recordings": int(quality["status"].ne("ready").sum()),
                "smoke_test": bool(args.smoke_test),
            },
            handle,
            indent=2,
        )
    with open(
        os.path.join(args.output_dir, "score_definition.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "transform": SCORE_TRANSFORM,
                "interpretation": {
                    "normal": "score < 0",
                    "boundary": "score == 0",
                    "abnormal": "score > 0",
                },
                "targets": SCORE_DEFINITIONS,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"Prepared synchronized data: requested_tasks={len(selected_targets)} "
        f"ready_tasks={len(ready_targets)} valid_recordings={len(recordings)} "
        f"excluded_recordings={quality['status'].ne('ready').sum()}",
        flush=True,
    )
    for row in task_summary.loc[
        task_summary["target"].isin(ready_targets)
    ].itertuples(index=False):
        print(
            f"[task-data] target={row.target} videos={int(row.synchronized_videos)} "
            f"windows={int(row.synchronized_windows)} "
            f"train/val/test={int(row.train_windows)}/"
            f"{int(row.val_windows)}/{int(row.test_windows)}",
            flush=True,
        )
    if args.prepare_only:
        print("Preparation complete; training was not started", flush=True)
        return

    available_gpus = torch.cuda.device_count()
    worker_count = min(
        args.workers or available_gpus, available_gpus, len(ready_targets)
    )
    if worker_count < 1:
        raise RuntimeError("At least one CUDA worker is required")
    jobs = [
        {
            "target": target,
            "windows_path": os.path.join(
                args.output_dir, "windows", f"{target}.csv"
            ),
            "run_dir": os.path.join(args.output_dir, "runs", target),
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "max_batches": args.max_batches,
        }
        for target in ready_targets
    ]
    print(
        f"Dynamic scheduler: jobs={len(jobs)} workers={worker_count} "
        f"available_gpus={available_gpus} assignments="
        f"{','.join(f'cuda:{gpu}' for gpu in range(worker_count))}",
        flush=True,
    )

    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in range(worker_count):
        gpu_queue.put(gpu_id)
    run_rows, metric_frames, failures = [], [], []
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_worker_init,
            initargs=(gpu_queue,),
        ) as executor:
            futures = {executor.submit(_worker_train, job): job for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                target = job["target"]
                try:
                    result = future.result()
                    metric_frames.append(result["metrics"])
                    status, reason, job_seed = "ok", "", result["job_seed"]
                except Exception as exc:
                    status, reason, job_seed = "failed", str(exc), np.nan
                    os.makedirs(job["run_dir"], exist_ok=True)
                    with open(
                        os.path.join(job["run_dir"], "error.txt"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(traceback.format_exc())
                    failures.append({"target": target, "error": str(exc)})
                    print(f"[job-failed] target={target}: {exc}", flush=True)
                run_rows.append(
                    {
                        "target": target,
                        "status": status,
                        "reason": reason,
                        "job_seed": job_seed,
                        "run_dir": job["run_dir"],
                    }
                )
                pd.DataFrame(run_rows).to_csv(run_index_path, index=False)
                if metric_frames:
                    pd.concat(metric_frames, ignore_index=True).to_csv(
                        os.path.join(args.output_dir, "metrics_all.csv"), index=False
                    )
                _collect_histories(args.output_dir).to_csv(
                    os.path.join(args.output_dir, "history_all.csv"), index=False
                )
                print(
                    f"[scheduler] {completed}/{len(jobs)} target={target} "
                    f"status={status}",
                    flush=True,
                )
    finally:
        manager.shutdown()
    pd.DataFrame(failures, columns=("target", "error")).to_csv(
        os.path.join(args.output_dir, "failures.csv"), index=False
    )
    print(f"Experiment outputs saved to {args.output_dir}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; see failures.csv")


if __name__ == "__main__":
    main()
