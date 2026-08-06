"""Run the controlled video-only Mamba ablation after parent-data validation."""

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

from study.exp2_video_ecg_mamba.models import dependency_versions

from .config import (
    D_CONV,
    D_MODEL,
    D_STATE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EXPAND,
    HEAD_HIDDEN_FEATURES,
    LEARNING_RATE,
    MAMBA_LAYERS,
    MAMBA_SSM_VERSION,
    MAX_EPOCHS,
    OUTPUT_DIR,
    PAIRED_JOB_SEED_TOKEN,
    PARENT_OUTPUT_DIR,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    TARGETS,
    TRAIN_BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WEIGHT_DECAY,
    WINDOW_SECONDS,
)
from .data import prepare_ablation_data
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


def _paired_job_seed(seed, target):
    token = f"{seed}:{target}:{PAIRED_JOB_SEED_TOKEN}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return (int(seed) + offset) % (2**31 - 1)


def _worker_train(job):
    job_seed = _paired_job_seed(job["seed"], job["target"])
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
            f"found {mamba_ssm.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Official mamba-ssm CUDA kernels require an NVIDIA GPU")
    model = mamba_ssm.Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda()
    inputs = torch.randn(1, 32, 32, device="cuda", requires_grad=True)
    model(inputs).mean().backward()
    if not torch.isfinite(inputs.grad).all():
        raise RuntimeError("mamba-ssm CUDA backward validation failed")
    del model, inputs
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


def _validate_controlled_hyperparameters(parent_output_dir):
    path = os.path.join(parent_output_dir, "experiment_manifest.json")
    with open(path, encoding="utf-8") as handle:
        parent = json.load(handle)
    expected_model = {
        "d_model": D_MODEL,
        "d_state": D_STATE,
        "d_conv": D_CONV,
        "expand": EXPAND,
        "layers": MAMBA_LAYERS,
        "head_hidden_features": HEAD_HIDDEN_FEATURES,
        "dropout": DROPOUT,
    }
    expected_training = {
        "batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "train_data_workers_per_job": TRAIN_NUM_WORKERS,
        "eval_data_workers_per_job": EVAL_NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }
    mismatches = []
    for key, expected in expected_model.items():
        actual = parent.get("model", {}).get(key)
        if actual != expected:
            mismatches.append(f"model.{key}: parent={actual}, ablation={expected}")
    for key, expected in expected_training.items():
        actual = parent.get("training", {}).get(key)
        if actual != expected:
            mismatches.append(f"training.{key}: parent={actual}, ablation={expected}")
    objective = parent.get("training", {}).get("objective", {})
    if objective.get("name") != "SmoothL1Loss" or not np.isclose(
        float(objective.get("beta", np.nan)), SMOOTH_L1_BETA
    ):
        mismatches.append(f"training.objective: parent={objective}")
    if mismatches:
        raise RuntimeError(
            "Controlled ablation hyperparameters differ:\n" + "\n".join(mismatches)
        )
    return parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-output-dir", default=PARENT_OUTPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-incomplete-parent", action="store_true")
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
    parent = _validate_controlled_hyperparameters(args.parent_output_dir)
    task_frames, hash_frame = prepare_ablation_data(
        parent_output_dir=args.parent_output_dir,
        output_dir=args.output_dir,
        targets=selected_targets,
        require_parent_complete=not args.allow_incomplete_parent,
    )
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "experiment": "native_video_only_mamba_ablation",
                "ablation": "remove_ecg_input_only",
                "paired_parent_output": os.path.abspath(args.parent_output_dir),
                "task_type": "abnormal_score_regression",
                "targets": selected_targets,
                "base_seed": int(args.seed),
                "job_seed_derivation": (
                    f"identical SHA256 token suffix: {PAIRED_JOB_SEED_TOKEN}"
                ),
                "controlled_variables": {
                    "exact_window_rows_and_order": True,
                    "patient_splits": True,
                    "target_scores": True,
                    "video_frame_boundaries": True,
                    "native_video_size": [3, VIDEO_HEIGHT, VIDEO_WIDTH],
                    "video_frame_sampling": False,
                    "video_interpolation": False,
                    "video_augmentation": "temporally consistent horizontal flip",
                    "shared_parameter_initialization": True,
                    "mamba_and_head_architecture": True,
                    "training_hyperparameters": True,
                    "loss_and_checkpoint_selection": True,
                },
                "changed_variable": {
                    "parent": "native video tokens + 512 Hz ECG tokens",
                    "ablation": "native video tokens only",
                },
                "model": parent["model"],
                "training": {
                    **parent["training"],
                    "max_epochs": int(args.max_epochs),
                    "early_stopping_patience": int(args.patience),
                },
                "score_transform": SCORE_TRANSFORM,
                "dependencies": dependency_versions(),
                "window_hashes": hash_frame[
                    ["target", "sha256", "rows", "videos"]
                ].to_dict("records"),
                "smoke_test": bool(args.smoke_test),
                "incomplete_parent_allowed": bool(args.allow_incomplete_parent),
            },
            handle,
            indent=2,
        )
    print(
        f"Prepared exact parent-window snapshot: tasks={len(task_frames)} "
        f"windows={sum(len(frame) for frame in task_frames.values())}",
        flush=True,
    )
    for target, frame in task_frames.items():
        counts = frame["split"].value_counts()
        print(
            f"[task-data] target={target} videos={frame['video_id'].nunique()} "
            f"windows={len(frame)} train/val/test="
            f"{counts.get('train', 0)}/{counts.get('val', 0)}/"
            f"{counts.get('test', 0)} sha256="
            f"{hash_frame.loc[hash_frame['target'].eq(target), 'sha256'].iloc[0]}",
            flush=True,
        )
    if args.prepare_only:
        print("Preparation complete; training was not started", flush=True)
        return

    available_gpus = torch.cuda.device_count()
    worker_count = min(
        args.workers or available_gpus, available_gpus, len(selected_targets)
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
        for target in selected_targets
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
    print(f"Ablation outputs saved to {args.output_dir}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; see failures.csv")


if __name__ == "__main__":
    main()
