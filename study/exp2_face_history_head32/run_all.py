"""Run 20-frame binary classification with prior lab history."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import threading
import time
import traceback

import numpy as np
import pandas as pd
import torch

from .config import (
    ARCHITECTURES,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    FRAMES_PER_VIDEO,
    FRAME_SHUFFLE_CHUNK_SIZE,
    GPU_AVAILABILITY_POLL_SECONDS,
    GPU_FREE_MEMORY_USED_MIB_MAX,
    HEAD_HIDDEN_FEATURES,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    HISTORY_HIDDEN_FEATURES,
    HISTORY_INPUT_FEATURES,
    HISTORY_OUTPUT_FEATURES,
    HISTORY_POLICY,
    EVAL_BATCH_SIZES,
    EVAL_NUM_WORKERS,
    JPEG_DECODER,
    OUTPUT_DIRS,
    REFERENCE_INDEX_DIR,
    REFERENCE_OUTPUT_DIR,
    REFERENCE_SOURCE_DATA_DIR,
    SOURCE_DATA_DIR,
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SEED,
    TARGETS,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZES,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    VIEW_NAMES,
    WEIGHTS_DIR,
)
from .data import prepare_tasks, validate_source_data
from .frame_index import FrameOffsetIndex, build_or_reuse_frame_index
from .history_data import (
    HistoryFeatureStore,
    build_history_artifacts,
    write_history_manifest,
)
from .models import HistoryEncoder, WEIGHT_FILES
from .source_data import build_raw_video_source
from .train import train_task


LAB_TARGET_PREFIXES = {
    "hemoglobin_low": "hemoglobin",
    "po2_low": "po2",
    "lactate_high": "lactate",
}

_WORKER_FRAME_INDEX = None
_WORKER_GPU_ID = None


def _parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


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


def _gpu_memory_used_mib():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = {}
    for line in completed.stdout.splitlines():
        index, memory = (part.strip() for part in line.split(",", maxsplit=1))
        result[int(index)] = int(memory)
    return result


def _monitor_gpu_admission(gpu_queue, candidate_gpus, admitted, lock, stop_event):
    while not stop_event.is_set():
        try:
            memory = _gpu_memory_used_mib()
        except Exception as exc:
            print(f"[gpu-monitor-warning] {type(exc).__name__}: {exc}", flush=True)
            stop_event.wait(GPU_AVAILABILITY_POLL_SECONDS)
            continue
        with lock:
            newly_free = [
                gpu_id
                for gpu_id in candidate_gpus
                if gpu_id not in admitted
                and memory.get(gpu_id, np.inf) <= GPU_FREE_MEMORY_USED_MIB_MAX
            ]
            for gpu_id in newly_free:
                admitted.add(gpu_id)
                gpu_queue.put(gpu_id)
                print(
                    f"[gpu-admitted] cuda:{gpu_id} memory_used={memory[gpu_id]}MiB",
                    flush=True,
                )
        stop_event.wait(GPU_AVAILABILITY_POLL_SECONDS)


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
    token = f"{job['seed']}:{job['architecture']}:{job['target']}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    job_seed = (job["seed"] + offset) % (2**31 - 1)
    random.seed(job_seed)
    np.random.seed(job_seed)
    torch.manual_seed(job_seed)
    torch.cuda.manual_seed_all(job_seed)
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str})
    metrics = train_task(
        architecture=job["architecture"],
        target=job["target"],
        frame_index=_WORKER_FRAME_INDEX,
        records=records,
        history_path=job["history_path"],
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
        "target": job["target"],
        "run_dir": job["run_dir"],
        "job_seed": job_seed,
        "metrics": metrics,
    }


def _validate_time_alignment(manifest, targets):
    for target in targets:
        prefix = LAB_TARGET_PREFIXES.get(target)
        if prefix is None:
            continue
        labelled = pd.to_numeric(manifest[target], errors="coerce").notna()
        delta = pd.to_numeric(manifest.loc[labelled, f"{prefix}_delta_h"], errors="coerce")
        signed = pd.to_numeric(
            manifest.loc[labelled, f"{prefix}_signed_delta_h"], errors="coerce"
        )
        invalid = (
            delta.isna()
            | signed.isna()
            | delta.lt(0.0)
            | delta.gt(24.0 + 1e-6)
            | ~np.isclose(delta.to_numpy(), np.abs(signed.to_numpy()), atol=1e-7)
        )
        if invalid.any():
            raise ValueError(f"Invalid 24-hour alignment for {target}: {int(invalid.sum())}")


def _validate_weights(weights_dir, architectures):
    missing = [
        os.path.join(weights_dir, WEIGHT_FILES[name])
        for name in architectures
        if not os.path.exists(os.path.join(weights_dir, WEIGHT_FILES[name]))
    ]
    if missing:
        raise FileNotFoundError(f"Missing local pretrained weights: {missing}")
    manifest_path = os.path.join(weights_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing weight manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def _collect_histories(output_dir):
    frames = []
    for root, _, files in os.walk(os.path.join(output_dir, "runs")):
        if "history.csv" in files:
            frames.append(pd.read_csv(os.path.join(root, "history.csv")))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _add_frame_counts(summary, task_records, frame_index):
    result = summary.copy()
    for target, records in task_records.items():
        for split in ("train", "val", "test"):
            count = 0
            for video_id in records.loc[records["split"].eq(split), "video_id"]:
                start, end = frame_index.frame_range(video_id)
                count += end - start
            result.loc[result["target"].eq(target), f"{split}_source_frames"] = count
        train_frames = int(
            result.loc[result["target"].eq(target), "train_source_frames"].iloc[0]
        )
        result.loc[result["target"].eq(target), "train_augmented_inputs"] = (
            train_frames * len(VIEW_NAMES)
        )
    return result


def _validate_reference_task_records(task_records, reference_output_dir):
    """Require identical samples/splits while auditing canonical value changes."""
    for target, records in task_records.items():
        path = os.path.join(reference_output_dir, "task_records", f"{target}.csv")
        reference = pd.read_csv(path, dtype={"hospital_id": str})
        current = records.sort_values("video_id").reset_index(drop=True)
        reference = reference.sort_values("video_id").reset_index(drop=True)
        stable_columns = [
            "hospital_id",
            "video_id",
            "mirror",
            "lab_patient_id",
            "binary_label",
            "source_sample_id",
            "split",
        ]
        pd.testing.assert_frame_equal(
            current[stable_columns],
            reference[stable_columns],
            check_dtype=False,
            check_exact=True,
        )
        changed_values = int(
            np.count_nonzero(
                ~np.isclose(
                current["raw_value"].to_numpy(np.float64),
                reference["raw_value"].to_numpy(np.float64),
                rtol=0.0,
                atol=1e-14,
            )
            )
        )
        print(
            f"[data-match] target={target} videos={len(current)} "
            "samples_labels_and_split=exact "
            f"canonical_raw_values_changed={changed_values}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-policy",
        choices=tuple(OUTPUT_DIRS),
        default="20frame",
    )
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--weights-dir", default=WEIGHTS_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--reference-output-dir", default=REFERENCE_OUTPUT_DIR)
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--targets", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--head-epochs", type=int, default=HEAD_MAX_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_MAX_EPOCHS)
    parser.add_argument("--head-patience", type=int, default=HEAD_PATIENCE)
    parser.add_argument("--finetune-patience", type=int, default=FINETUNE_PATIENCE)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir or OUTPUT_DIRS[args.frame_policy]
    args.source_dir = args.source_dir or SOURCE_DATA_DIR
    args.index_dir = args.index_dir or REFERENCE_INDEX_DIR

    architectures = _parse_csv(args.architectures)
    requested_targets = (
        _parse_csv(args.targets)
        if args.targets
        else TARGETS
    )
    unknown_architectures = sorted(set(architectures) - set(ARCHITECTURES))
    unknown_targets = sorted(set(requested_targets) - set(TARGETS))
    if unknown_architectures or unknown_targets:
        raise ValueError(
            f"Unknown architectures={unknown_architectures}, targets={unknown_targets}"
        )
    if args.smoke_test:
        args.head_epochs = args.finetune_epochs = 1
        args.head_patience = args.finetune_patience = 1
        args.max_batches = args.max_batches or 2

    run_index_path = os.path.join(args.output_dir, "run_index.csv")
    if os.path.exists(run_index_path) and not args.overwrite:
        raise FileExistsError(
            f"Output already contains a run: {args.output_dir}. Use --overwrite explicitly."
        )
    if args.overwrite and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)
    _set_seed(args.seed)
    build_raw_video_source(args.source_dir, requested_targets)
    source_quality = validate_source_data(args.source_dir)
    weight_manifest = _validate_weights(args.weights_dir, architectures)
    base_manifest = pd.read_csv(
        os.path.join(args.source_dir, "base_manifest.csv"), dtype={"hospital_id": str}
    )
    video_summary = pd.read_csv(
        os.path.join(args.source_dir, "video_summary.csv"),
        dtype={"hospital_id": str},
    )
    _validate_time_alignment(base_manifest, requested_targets)
    candidate_mask = base_manifest[list(requested_targets)].notna().any(axis=1)
    candidate_videos = base_manifest.loc[candidate_mask].drop_duplicates("video_id")
    frame_index = build_or_reuse_frame_index(
        candidate_videos,
        args.index_dir,
        frame_policy=args.frame_policy,
    )
    indexed_video_ids = set(frame_index.video_lookup)
    excluded_frame_videos = set(
        candidate_videos["video_id"].astype(str)
    ) - indexed_video_ids
    if excluded_frame_videos:
        eligibility = (
            f"{FRAMES_PER_VIDEO} non-adjacent decodable frames"
            if args.frame_policy == "20frame"
            else "any decodable frame"
        )
        print(
            f"Excluded {len(excluded_frame_videos)} labelled videos that cannot "
            f"provide {eligibility}",
            flush=True,
        )
    base_manifest = base_manifest[
        base_manifest["video_id"].astype(str).isin(indexed_video_ids)
    ].reset_index(drop=True)
    video_summary = video_summary[
        video_summary["video_id"].astype(str).isin(indexed_video_ids)
    ].reset_index(drop=True)
    task_records, task_summary, _ = prepare_tasks(
        base_manifest,
        video_summary,
        args.output_dir,
        requested_targets,
        args.seed,
        reference_records_dir=os.path.join(
            args.reference_output_dir, "task_records"
        ),
    )
    ready_targets = tuple(
        task_summary.loc[task_summary["status"].eq("ready"), "target"].astype(str)
    )
    if set(ready_targets) != set(requested_targets):
        skipped = task_summary.loc[~task_summary["status"].eq("ready"), ["target", "reason"]]
        raise RuntimeError(
            f"Requested {args.frame_policy} tasks are not trainable: "
            f"{skipped.to_dict('records')}"
        )
    _validate_reference_task_records(task_records, args.reference_output_dir)
    history_dir = os.path.join(args.output_dir, "history_records")
    history_summaries = [
        build_history_artifacts(
            target,
            task_records[target],
            base_manifest,
            history_dir,
        )
        for target in ready_targets
    ]
    write_history_manifest(history_summaries, history_dir)
    with open(
        os.path.join(args.output_dir, "split_assignment_manifest.json"),
        encoding="utf-8",
    ) as handle:
        split_assignment_manifest = json.load(handle)
    if args.smoke_test:
        ready_targets = ready_targets[:1]

    union_videos = pd.concat(
        [task_records[target] for target in ready_targets], ignore_index=True
    ).drop_duplicates("video_id")
    task_summary = _add_frame_counts(task_summary, task_records, frame_index)
    task_summary.to_csv(os.path.join(args.output_dir, "task_summary.csv"), index=False)

    index_manifest_path = os.path.join(args.index_dir, "index_manifest.json")
    with open(index_manifest_path, encoding="utf-8") as handle:
        index_manifest = json.load(handle)
    experiment_manifest = {
        "schema_version": 1,
        "experiment": "exp2_face_history_head32_binary_classification",
        "result_variant": args.frame_policy,
        "source_dir": os.path.abspath(args.source_dir),
        "source_data_quality_report": source_quality,
        "architectures": list(architectures),
        "targets": list(ready_targets),
        "seed": args.seed,
        "data_fingerprints": {
            "base_manifest_sha256": _sha256_file(
                os.path.join(args.source_dir, "base_manifest.csv")
            ),
            "data_quality_report_sha256": _sha256_file(
                os.path.join(args.source_dir, "data_quality_report.json")
            ),
            "task_record_sha256": {
                target: _sha256_file(
                    os.path.join(args.output_dir, "task_records", f"{target}.csv")
                )
                for target in ready_targets
            },
            "history_record_sha256": {
                target: _sha256_file(
                    os.path.join(history_dir, f"{target}.csv")
                )
                for target in ready_targets
            },
            "history_feature_sha256": {
                target: _sha256_file(
                    os.path.join(history_dir, f"{target}.npz")
                )
                for target in ready_targets
            },
            "split_distribution_audit_sha256": _sha256_file(
                os.path.join(args.output_dir, "split_distribution_audit.csv")
            ),
            "split_distribution_pairwise_sha256": _sha256_file(
                os.path.join(args.output_dir, "split_distribution_pairwise.csv")
            ),
            "split_assignment_manifest_sha256": _sha256_file(
                os.path.join(args.output_dir, "split_assignment_manifest.json")
            ),
        },
        "model_head": {
            "type": "concat(image_features, history_features)-Linear-LayerNorm-SiLU-Dropout-Linear",
            "hidden_features": HEAD_HIDDEN_FEATURES,
            "dropout": 0.25,
            "output_activation": None,
        },
        "history_encoder": {
            "type": "per-measurement MLP with masked mean pooling",
            "input_features": HISTORY_INPUT_FEATURES,
            "hidden_features": HISTORY_HIDDEN_FEATURES,
            "output_features": HISTORY_OUTPUT_FEATURES,
            "parameter_count": sum(
                parameter.numel() for parameter in HistoryEncoder().parameters()
            ),
            "policy": HISTORY_POLICY,
            "artifacts_dir": os.path.abspath(history_dir),
        },
        "classification_target": {
            "name": "binary_label",
            "definitions": SCORE_DEFINITIONS,
            "duplicate_event_policy": (
                "one nearest in-window lab measurement per raw video and target"
            ),
            "po2_item_policy": (
                "use exact item_name '氧分压'; exclude all patient-temperature-"
                "corrected PO2 rows"
            ),
        },
        "preprocessing": {
            "frame_policy": (
                (
                    f"{FRAMES_PER_VIDEO} deterministic non-adjacent RGB MJPEG "
                    "frames per video"
                )
                if args.frame_policy == "20frame"
                else "all decodable RGB MJPEG frames per video"
            ),
            "training_views": list(VIEW_NAMES),
            "model_input_shape": [3, 224, 224],
            "normalization": "ImageNet mean/std",
            "conflict_policy": (
                "not applicable after nearest-measurement selection; one label "
                "per video and target"
            ),
            "frame_eligibility_policy": (
                (
                    f"exclude videos that cannot provide {FRAMES_PER_VIDEO} "
                    "unique decodable frames at the configured non-adjacent spacing"
                )
                if args.frame_policy == "20frame"
                else "exclude videos with no decodable 128x128 RGB MJPEG frames"
            ),
            "frame_ineligible_labelled_videos": sorted(excluded_frame_videos),
            "split_policy": (
                "patient-disjoint 60/20/20 class-stratified candidate search; "
                "selected by video-level raw-value and abnormal-score distribution"
            ),
            "evaluation_unit": (
                (
                    f"video; mean predicted probability over {FRAMES_PER_VIDEO} "
                    "selected source frames"
                )
                if args.frame_policy == "20frame"
                else "video; mean predicted abnormal score over all indexed frames"
            ),
        },
        "split_assignment": split_assignment_manifest,
        "storage_and_io": {
            "decoded_frame_cache_on_disk": False,
            "index_path": os.path.abspath(args.index_dir),
            "index_size_bytes": os.path.getsize(os.path.join(args.index_dir, "frame_offsets.npz")),
            "total_indexed_frames": index_manifest["total_valid_frames"],
            "training_decode_policy": (
                "persistent file handles + byte seek + bounded RAM LRU; "
                "only indexed frame offsets are decoded; "
                "views/resize/normalization run on GPU"
            ),
            "jpeg_decoder": JPEG_DECODER,
            "jpeg_decoder_equivalence": "pixel-exact against prior PIL RGB path",
            "training_frame_shuffle_chunk_size": FRAME_SHUFFLE_CHUNK_SIZE,
            "frame_prediction_format": "compressed numeric NPZ; no per-frame CSV",
        },
        "training": {
            "stage_1": (
                f"frozen image encoder; train history encoder and classification head, "
                f"lr={HEAD_LEARNING_RATE:.8g}"
            ),
            "stage_2": (
                f"all parameters unfrozen, lr={FINETUNE_LEARNING_RATE:.8g}"
            ),
            "objective": "BCEWithLogitsLoss",
            "class_weight_basis": "actual valid training-frame counts",
            "head_max_epochs": args.head_epochs,
            "finetune_max_epochs": args.finetune_epochs,
            "scheduler": (
                "GPU-memory-aware dynamic process queue; idle GPUs admitted at start "
                "and busy GPUs admitted after release"
            ),
            "workers_per_gpu": args.workers_per_gpu,
            "explicit_worker_override": args.workers,
            "train_source_batch_sizes": TRAIN_SOURCE_BATCH_SIZES,
            "effective_train_batch_sizes": {
                name: size * len(VIEW_NAMES)
                for name, size in TRAIN_SOURCE_BATCH_SIZES.items()
            },
            "eval_batch_sizes": EVAL_BATCH_SIZES,
            "train_decode_workers_per_job": TRAIN_NUM_WORKERS,
            "eval_decode_workers_per_split": EVAL_NUM_WORKERS,
            "memory_format": "channels_last",
            "mixed_precision": "float16 autocast with GradScaler",
            "torch_compile_enabled": TORCH_COMPILE_ENABLED,
            "torch_compile_mode": TORCH_COMPILE_MODE,
            "checkpoint_state_source": "unwrapped eager model",
        },
        "pretrained_weights": weight_manifest,
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(experiment_manifest, handle, ensure_ascii=False, indent=2)
    print(
        f"Prepared {args.frame_policy} raw-video experiment: "
        f"videos={len(union_videos)} "
        f"indexed_frames={index_manifest['total_valid_frames']} "
        f"tasks={len(ready_targets)} architectures={len(architectures)}",
        flush=True,
    )
    if args.prepare_only:
        print("Preparation complete; training was not started", flush=True)
        return

    jobs = []
    for architecture in architectures:
        for target in ready_targets:
            jobs.append({
                "architecture": architecture,
                "target": target,
                "records_path": os.path.join(
                    args.output_dir, "task_records", f"{target}.csv"
                ),
                "history_path": os.path.join(
                    history_dir, f"{target}.npz"
                ),
                "weights_dir": args.weights_dir,
                "run_dir": os.path.join(args.output_dir, "runs", architecture, target),
                "seed": args.seed,
                "head_epochs": args.head_epochs,
                "finetune_epochs": args.finetune_epochs,
                "head_patience": args.head_patience,
                "finetune_patience": args.finetune_patience,
                "max_batches": args.max_batches,
            })
    available_gpus = torch.cuda.device_count()
    if available_gpus < 1:
        raise RuntimeError(f"The {args.frame_policy} experiment requires CUDA")
    requested_workers = (
        args.workers
        if args.workers is not None
        else available_gpus * args.workers_per_gpu
    )
    worker_count = min(requested_workers, len(jobs))
    if worker_count < 1:
        raise ValueError(
            f"Invalid worker configuration: workers={args.workers}, "
            f"workers_per_gpu={args.workers_per_gpu}"
        )
    candidate_gpus = list(range(available_gpus))

    run_rows, metric_frames, failure_rows = [], [], []
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    admitted = set()
    admission_lock = threading.Lock()
    initial_memory = _gpu_memory_used_mib()
    for gpu_id in candidate_gpus:
        if initial_memory.get(gpu_id, np.inf) <= GPU_FREE_MEMORY_USED_MIB_MAX:
            admitted.add(gpu_id)
            gpu_queue.put(gpu_id)
    print(
        f"Dynamic GPU scheduler: jobs={len(jobs)} workers={worker_count} "
        f"candidate_gpus={candidate_gpus} initially_admitted={sorted(admitted)} "
        f"held_busy={sorted(set(candidate_gpus) - admitted)} "
        f"memory_threshold={GPU_FREE_MEMORY_USED_MIB_MAX}MiB",
        flush=True,
    )
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_monitor_gpu_admission,
        args=(
            gpu_queue,
            candidate_gpus,
            admitted,
            admission_lock,
            stop_monitor,
        ),
        name="gpu-admission-monitor",
        daemon=True,
    )
    monitor.start()
    index_path = os.path.join(args.index_dir, "frame_offsets.npz")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_worker_init,
            initargs=(index_path, gpu_queue),
        ) as executor:
            try:
                futures = {executor.submit(_worker_train, job): job for job in jobs}
                for completed, future in enumerate(as_completed(futures), start=1):
                    job = futures[future]
                    architecture, target, run_dir = (
                        job["architecture"], job["target"], job["run_dir"]
                    )
                    try:
                        result = future.result()
                        metric_frames.append(result["metrics"])
                        status, reason, job_seed = "ok", "", result["job_seed"]
                    except Exception as exc:
                        status, reason, job_seed = "failed", str(exc), np.nan
                        os.makedirs(run_dir, exist_ok=True)
                        with open(
                            os.path.join(run_dir, "error.txt"), "w", encoding="utf-8"
                        ) as handle:
                            handle.write(traceback.format_exc())
                        failure_rows.append({
                            "architecture": architecture,
                            "target": target,
                            "error": str(exc),
                        })
                        print(
                            f"[job-failed] arch={architecture} task={target}: {exc}",
                            flush=True,
                        )
                    run_rows.append({
                        "architecture": architecture,
                        "target": target,
                        "status": status,
                        "reason": reason,
                        "job_seed": job_seed,
                        "run_dir": run_dir,
                    })
                    pd.DataFrame(run_rows).to_csv(run_index_path, index=False)
                    if metric_frames:
                        pd.concat(metric_frames, ignore_index=True).to_csv(
                            os.path.join(args.output_dir, "metrics_all.csv"), index=False
                        )
                    _collect_histories(args.output_dir).to_csv(
                        os.path.join(args.output_dir, "history_all.csv"), index=False
                    )
                    print(
                        f"[scheduler] {completed}/{len(jobs)} "
                        f"arch={architecture} task={target} status={status}",
                        flush=True,
                    )
            finally:
                stop_monitor.set()
                # Release workers still blocked in their initializer. No jobs
                # remain for these fallback assignments.
                for _ in range(worker_count):
                    gpu_queue.put(0)
    finally:
        stop_monitor.set()
        monitor.join(timeout=GPU_AVAILABILITY_POLL_SECONDS + 2)
        manager.shutdown()
    pd.DataFrame(
        failure_rows, columns=("architecture", "target", "error")
    ).to_csv(os.path.join(args.output_dir, "failures.csv"), index=False)
    print(f"Experiment outputs saved to {args.output_dir}", flush=True)
    if failure_rows:
        raise RuntimeError(f"{len(failure_rows)} jobs failed; see failures.csv")
    print("[plot] Generating result figures", flush=True)
    from .plot_results import main as plot_results

    plot_results(args.output_dir)


if __name__ == "__main__":
    main()
