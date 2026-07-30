"""Prepare data once and dynamically schedule one five-output model per backbone."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.frame_index import (
    FrameOffsetIndex,
    build_or_reuse_frame_index,
)
from study.exp2_face_pretrained_head32_regression.run_all import (
    _validate_time_alignment,
)

from .config import (
    ARCHITECTURES,
    EVAL_BATCH_SIZES,
    EVAL_NUM_WORKERS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    FRAME_SHUFFLE_CHUNK_SIZE,
    HEAD_HIDDEN_FEATURES,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    JPEG_DECODER,
    OUTPUT_DIR,
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    SOURCE_DATA_DIR,
    SPLIT_CANDIDATES,
    TARGETS,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZES,
    VIEW_NAMES,
    WEIGHTS_DIR,
)
from .data import prepare_multitask_data, validate_source_data
from .models import WEIGHT_FILES
from .train import train_architecture


_WORKER_FRAME_INDEX = None
_WORKER_GPU_ID = None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _validate_weights(weights_dir, architectures):
    missing = [
        os.path.join(weights_dir, WEIGHT_FILES[name])
        for name in architectures
        if not os.path.isfile(os.path.join(weights_dir, WEIGHT_FILES[name]))
    ]
    if missing:
        raise FileNotFoundError(f"Missing local pretrained weights: {missing}")
    manifest_path = os.path.join(weights_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing weight manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def _worker_init(index_path, gpu_queue):
    global _WORKER_FRAME_INDEX, _WORKER_GPU_ID
    _WORKER_GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_WORKER_GPU_ID)
    torch.set_num_threads(1)
    _WORKER_FRAME_INDEX = FrameOffsetIndex.load(index_path)
    print(
        f"[worker-ready] pid={os.getpid()} gpu=cuda:{_WORKER_GPU_ID} "
        f"indexed_frames={len(_WORKER_FRAME_INDEX.starts)}",
        flush=True,
    )


def _worker_train(job):
    token = f"{job['seed']}:{job['architecture']}:multitask".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    job_seed = (job["seed"] + offset) % (2**31 - 1)
    random.seed(job_seed)
    np.random.seed(job_seed)
    torch.manual_seed(job_seed)
    torch.cuda.manual_seed_all(job_seed)
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str})
    metrics = train_architecture(
        architecture=job["architecture"],
        frame_index=_WORKER_FRAME_INDEX,
        records=records,
        task_weights=np.asarray(job["task_weights"], dtype=np.float32),
        task_frame_counts=np.asarray(job["task_frame_counts"], dtype=np.int64),
        weights_dir=job["weights_dir"],
        run_dir=job["run_dir"],
        head_epochs=job["head_epochs"],
        finetune_epochs=job["finetune_epochs"],
        head_patience=job["head_patience"],
        finetune_patience=job["finetune_patience"],
        max_batches=job["max_batches"],
    )
    return {
        "architecture": job["architecture"],
        "run_dir": job["run_dir"],
        "job_seed": job_seed,
        "metrics": metrics,
    }


def _frame_label_counts(records, frame_index):
    counts = np.zeros(len(TARGETS), dtype=np.int64)
    split_counts = {
        split: np.zeros(len(TARGETS), dtype=np.int64)
        for split in ("train", "val", "test")
    }
    for row in records.itertuples(index=False):
        start, end = frame_index.frame_range(row.video_id)
        frames = end - start
        masks = np.asarray(
            [getattr(row, f"{target}__mask") for target in TARGETS],
            dtype=np.int64,
        )
        split_counts[row.split] += frames * masks
    counts[:] = split_counts["train"]
    if (counts < 1).any():
        missing = [target for target, count in zip(TARGETS, counts) if count < 1]
        raise RuntimeError(f"No training frame labels for tasks: {missing}")
    weights = counts.mean(dtype=np.float64) / counts
    return split_counts, weights.astype(np.float32)


def _update_task_summary(summary, split_counts):
    result = summary.copy()
    for task_index, target in enumerate(TARGETS):
        for split in ("train", "val", "test"):
            result.loc[
                result["target"].eq(target), f"{split}_source_frames"
            ] = int(split_counts[split][task_index])
        result.loc[
            result["target"].eq(target), "train_augmented_inputs"
        ] = int(split_counts["train"][task_index] * len(VIEW_NAMES))
    return result


def _collect_histories(output_dir):
    frames = []
    for architecture in ARCHITECTURES:
        path = os.path.join(output_dir, "runs", architecture, "history.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=SOURCE_DATA_DIR)
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--index-dir",
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "exp2_face_pretrained_allframes_head32",
                "outputs",
                "frame_index",
            )
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--head-epochs", type=int, default=HEAD_MAX_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_MAX_EPOCHS)
    parser.add_argument("--head-patience", type=int, default=HEAD_PATIENCE)
    parser.add_argument("--finetune-patience", type=int, default=FINETUNE_PATIENCE)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.head_epochs = args.finetune_epochs = 1
        args.head_patience = args.finetune_patience = 1
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
    source_quality = validate_source_data(args.source_dir)
    weight_manifest = _validate_weights(args.weights_dir, ARCHITECTURES)
    base_manifest = pd.read_csv(
        os.path.join(args.source_dir, "base_manifest.csv"),
        dtype={"hospital_id": str},
    )
    video_summary = pd.read_csv(
        os.path.join(args.source_dir, "video_summary.csv"),
        dtype={"hospital_id": str},
    )
    _validate_time_alignment(base_manifest, TARGETS)
    records, task_records, task_summary, split_manifest = prepare_multitask_data(
        base_manifest, video_summary, args.output_dir, TARGETS, args.seed
    )
    frame_index = build_or_reuse_frame_index(records, args.index_dir)
    split_frame_counts, task_weights = _frame_label_counts(records, frame_index)
    task_summary = _update_task_summary(task_summary, split_frame_counts)
    task_summary["train_task_loss_weight"] = task_weights
    task_summary.to_csv(os.path.join(args.output_dir, "task_summary.csv"), index=False)
    with open(
        os.path.join(args.output_dir, "task_loss_weights.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "task_order": list(TARGETS),
                "basis": "actual observed source-frame labels in the training split",
                "formula": "mean(task_frame_counts) / task_frame_count",
                "abnormal_side_weighting": False,
                "train_frame_label_counts": dict(
                    zip(TARGETS, map(int, split_frame_counts["train"]))
                ),
                "weights": dict(zip(TARGETS, map(float, task_weights))),
            },
            handle,
            indent=2,
        )

    index_manifest_path = os.path.join(args.index_dir, "index_manifest.json")
    with open(index_manifest_path, encoding="utf-8") as handle:
        index_manifest = json.load(handle)
    experiment_manifest = {
        "schema_version": 1,
        "experiment": "exp2_face_pretrained_allframes_head32_multitask_regression",
        "architectures": list(ARCHITECTURES),
        "target_order": list(TARGETS),
        "seed": args.seed,
        "source_dir": os.path.abspath(args.source_dir),
        "source_data_quality_report": source_quality,
        "data_fingerprints": {
            "base_manifest_sha256": _sha256_file(
                os.path.join(args.source_dir, "base_manifest.csv")
            ),
            "data_quality_report_sha256": _sha256_file(
                os.path.join(args.source_dir, "data_quality_report.json")
            ),
            "multitask_records_sha256": _sha256_file(
                os.path.join(args.output_dir, "multitask_records.csv")
            ),
            "patient_split_sha256": _sha256_file(
                os.path.join(args.output_dir, "patient_split.csv")
            ),
        },
        "model": {
            "backbone": "unchanged local ImageNet-pretrained backbone",
            "head": "Linear-LayerNorm-SiLU-Dropout-Linear",
            "hidden_features": HEAD_HIDDEN_FEATURES,
            "outputs": len(TARGETS),
            "output_order": list(TARGETS),
        },
        "data": {
            "join": "outer join over per-target clean video records",
            "missing_target_policy": (
                "retain each video with at least one target; mask unavailable targets"
            ),
            "conflict_policy": (
                "exclude a video only for a target that has both binary signs"
            ),
            "duplicate_event_policy": (
                "for each target/video, retain the event with minimum absolute "
                "video-lab time delta"
            ),
            "split_policy": (
                "one global patient-disjoint 60/20/20 assignment optimized over "
                "all five tasks' raw-value and abnormal-score distributions"
            ),
            "split_search_candidates": SPLIT_CANDIDATES,
            "split_assignment": split_manifest,
            "frame_policy": "every decodable 128x128 RGB MJPEG frame",
            "training_views": list(VIEW_NAMES),
            "evaluation_views": ["original"],
            "model_input_shape": [3, 224, 224],
            "score_transform": SCORE_TRANSFORM,
            "score_definitions": SCORE_DEFINITIONS,
        },
        "training": {
            "objective": "masked task-balanced SmoothL1 abnormal-score regression",
            "smooth_l1_beta": SMOOTH_L1_BETA,
            "task_weight_basis": (
                "inverse actual observed training source-frame-label count, "
                "normalized to mean 1"
            ),
            "abnormal_side_weighting": False,
            "task_loss_weights": dict(zip(TARGETS, map(float, task_weights))),
            "stage_1": f"frozen encoder, lr={HEAD_LEARNING_RATE:.8g}",
            "stage_2": f"all parameters unfrozen, lr={FINETUNE_LEARNING_RATE:.8g}",
            "head_max_epochs": args.head_epochs,
            "finetune_max_epochs": args.finetune_epochs,
            "early_stopping_metric": "macro video-level validation MAE",
            "train_source_batch_sizes": TRAIN_SOURCE_BATCH_SIZES,
            "effective_train_batch_sizes": {
                architecture: batch_size * len(VIEW_NAMES)
                for architecture, batch_size in TRAIN_SOURCE_BATCH_SIZES.items()
            },
            "eval_batch_sizes": EVAL_BATCH_SIZES,
            "train_decode_workers_per_job": TRAIN_NUM_WORKERS,
            "eval_decode_workers_per_split": EVAL_NUM_WORKERS,
            "torch_compile_enabled": TORCH_COMPILE_ENABLED,
            "torch_compile_mode": TORCH_COMPILE_MODE,
        },
        "storage_and_io": {
            "decoded_frame_cache_on_disk": False,
            "index_path": os.path.abspath(args.index_dir),
            "index_size_bytes": os.path.getsize(
                os.path.join(args.index_dir, "frame_offsets.npz")
            ),
            "total_indexed_frames": index_manifest["total_valid_frames"],
            "jpeg_decoder": JPEG_DECODER,
            "training_frame_shuffle_chunk_size": FRAME_SHUFFLE_CHUNK_SIZE,
        },
        "pretrained_weights": weight_manifest,
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(experiment_manifest, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, "score_definition.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "transform": SCORE_TRANSFORM,
                "definitions": SCORE_DEFINITIONS,
                "task_order": list(TARGETS),
                "boundary_semantics": (
                    "negative=normal side, positive=abnormal side, zero=boundary"
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Prepared multi-output experiment: patients="
        f"{records['hospital_id'].nunique()} videos={len(records)} "
        f"indexed_frames={index_manifest['total_valid_frames']} "
        f"architectures={len(ARCHITECTURES)}",
        flush=True,
    )
    print(
        f"Training frame-label counts: "
        f"{dict(zip(TARGETS, map(int, split_frame_counts['train'])))}",
        flush=True,
    )
    print(
        f"Task loss weights: {dict(zip(TARGETS, np.round(task_weights, 6)))}",
        flush=True,
    )
    if args.prepare_only:
        print("Preparation complete; training was not started", flush=True)
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError("The all-frame experiment requires CUDA")
    jobs = [
        {
            "architecture": architecture,
            "records_path": os.path.join(args.output_dir, "multitask_records.csv"),
            "task_weights": task_weights.tolist(),
            "task_frame_counts": split_frame_counts["train"].tolist(),
            "weights_dir": args.weights_dir,
            "run_dir": os.path.join(args.output_dir, "runs", architecture),
            "seed": args.seed,
            "head_epochs": args.head_epochs,
            "finetune_epochs": args.finetune_epochs,
            "head_patience": args.head_patience,
            "finetune_patience": args.finetune_patience,
            "max_batches": args.max_batches,
        }
        for architecture in ARCHITECTURES
    ]
    worker_count = min(args.workers or available_gpus, len(jobs))
    if worker_count < 1:
        raise ValueError(f"Invalid worker count: {worker_count}")
    gpu_ids = list(range(worker_count))
    print(
        f"Dynamic scheduler: jobs={len(jobs)} workers={worker_count} "
        f"available_gpus={available_gpus} assignments="
        f"{','.join(f'cuda:{gpu}' for gpu in gpu_ids)}",
        flush=True,
    )

    run_rows, metric_frames, failures = [], [], []
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_worker_init,
            initargs=(
                os.path.join(args.index_dir, "frame_offsets.npz"),
                gpu_queue,
            ),
        ) as executor:
            futures = {
                executor.submit(_worker_train, job): job for job in jobs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                architecture = job["architecture"]
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
                    failures.append(
                        {"architecture": architecture, "error": str(exc)}
                    )
                    print(
                        f"[job-failed] arch={architecture}: {exc}", flush=True
                    )
                run_rows.append(
                    {
                        "architecture": architecture,
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
                    f"[scheduler] {completed}/{len(jobs)} arch={architecture} "
                    f"status={status}",
                    flush=True,
                )
    finally:
        manager.shutdown()
    pd.DataFrame(failures, columns=("architecture", "error")).to_csv(
        os.path.join(args.output_dir, "failures.csv"), index=False
    )
    print(f"Experiment outputs saved to {args.output_dir}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; see failures.csv")


if __name__ == "__main__":
    main()
