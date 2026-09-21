"""Run robust-scaled raw laboratory value face-regression jobs."""

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

from .config import (
    ARCHITECTURES,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    FRAMES_PER_VIDEO,
    FRAME_SHUFFLE_CHUNK_SIZE,
    HEAD_HIDDEN_FEATURES,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    EVAL_BATCH_SIZES,
    EVAL_NUM_WORKERS,
    JPEG_DECODER,
    OUTPUT_DIRS,
    REFERENCE_INDEX_DIR,
    REFERENCE_OUTPUT_DIR,
    REGRESSION_TARGET_COLUMN,
    REGRESSION_TARGET_TRANSFORM,
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
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
from .models import WEIGHT_FILES
from .source_data import (
    TARGET_ANALYTES,
    build_raw_video_source,
    validate_analyte_source_policies,
)
from .scaling import fit_robust_target_scaler, write_target_scalers
from .train import train_task


LAB_TARGET_PREFIXES = TARGET_ANALYTES

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


def _validate_saved_checkpoint(run_dir, architecture, target):
    path = os.path.join(run_dir, "model.pt")
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RuntimeError(f"Missing or empty final checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != architecture:
        raise RuntimeError(f"Checkpoint architecture mismatch: {path}")
    if checkpoint.get("target") != target:
        raise RuntimeError(f"Checkpoint target mismatch: {path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no model_state_dict: {path}")
    if checkpoint.get("task_type") != "robust_scaled_raw_value_regression":
        raise RuntimeError(f"Checkpoint has the wrong regression target type: {path}")
    scaler = checkpoint.get("target_scaler")
    if not isinstance(scaler, dict) or scaler.get("target") != target:
        raise RuntimeError(f"Checkpoint has no matching target scaler: {path}")
    return {
        "model_pt_bytes": os.path.getsize(path),
        "model_pt_sha256": _sha256_file(path),
        "selected_stage": checkpoint.get("selected_stage", ""),
    }


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


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
        target_scaler=job["target_scaler"],
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
    for target, records in task_records.items():
        path = os.path.join(reference_output_dir, "task_records", f"{target}.csv")
        if not os.path.isfile(path):
            print(
                f"[data-match] target={target} reference=unavailable "
                "action=new_patient_disjoint_balanced_split",
                flush=True,
            )
            continue
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
        if not np.allclose(
            current["raw_value"].to_numpy(np.float64),
            reference["raw_value"].to_numpy(np.float64),
            rtol=0.0,
            atol=1e-14,
        ):
            raise AssertionError(f"Raw values differ from reference for {target}")
        print(
            f"[data-match] target={target} videos={len(current)} "
            "samples_labels_raw_values_and_split=exact",
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
    parser.add_argument("--reference-output-dir", default=None)
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
    parser.add_argument(
        "--add-targets",
        action="store_true",
        help="Train only --targets and preserve completed targets in this output.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir or OUTPUT_DIRS[args.frame_policy]
    args.source_dir = args.source_dir or os.path.join(args.output_dir, "source_data")
    args.index_dir = args.index_dir or (
        REFERENCE_INDEX_DIR
        if args.frame_policy == "20frame"
        else os.path.join(args.output_dir, "frame_index")
    )
    if args.reference_output_dir is None and args.frame_policy == "20frame":
        args.reference_output_dir = REFERENCE_OUTPUT_DIR

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
    if args.overwrite and args.add_targets:
        raise ValueError("--overwrite and --add-targets are mutually exclusive")
    if args.smoke_test:
        args.head_epochs = args.finetune_epochs = 1
        args.head_patience = args.finetune_patience = 1
        args.max_batches = args.max_batches or 2

    run_index_path = os.path.join(args.output_dir, "run_index.csv")
    existing_manifest = {}
    if args.add_targets:
        manifest_path = os.path.join(args.output_dir, "experiment_manifest.json")
        if not os.path.isfile(run_index_path) or not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                "--add-targets requires a completed run_index.csv and manifest"
            )
        with open(manifest_path, encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        existing_targets = tuple(existing_manifest.get("targets", ()))
        preparation_targets = tuple(
            target
            for target in TARGETS
            if target in set(existing_targets) | set(requested_targets)
        )
        manifest_architectures = tuple(
            architecture
            for architecture in ARCHITECTURES
            if architecture
            in set(existing_manifest.get("architectures", ())) | set(architectures)
        )
    else:
        preparation_targets = requested_targets
        manifest_architectures = architectures
    if os.path.exists(run_index_path) and not (args.overwrite or args.add_targets):
        raise FileExistsError(
            f"Output already contains a run: {args.output_dir}. Use --overwrite explicitly."
        )
    if args.overwrite and os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)
    _set_seed(args.seed)
    build_raw_video_source(args.source_dir, preparation_targets)
    source_quality = validate_source_data(args.source_dir)
    validate_analyte_source_policies(source_quality, preparation_targets)
    weight_manifest = _validate_weights(args.weights_dir, architectures)
    base_manifest = pd.read_csv(
        os.path.join(args.source_dir, "base_manifest.csv"), dtype={"hospital_id": str}
    )
    video_summary = pd.read_csv(
        os.path.join(args.source_dir, "video_summary.csv"),
        dtype={"hospital_id": str},
    )
    _validate_time_alignment(base_manifest, preparation_targets)
    candidate_mask = base_manifest[list(preparation_targets)].notna().any(axis=1)
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
        preparation_targets,
        args.seed,
        reference_records_dir=(
            os.path.join(args.reference_output_dir, "task_records")
            if args.reference_output_dir
            else None
        ),
    )
    ready_targets = tuple(
        task_summary.loc[task_summary["status"].eq("ready"), "target"].astype(str)
    )
    if set(ready_targets) != set(preparation_targets):
        skipped = task_summary.loc[~task_summary["status"].eq("ready"), ["target", "reason"]]
        raise RuntimeError(
            f"Requested {args.frame_policy} tasks are not trainable: "
            f"{skipped.to_dict('records')}"
        )
    if args.reference_output_dir:
        _validate_reference_task_records(task_records, args.reference_output_dir)
    target_scalers = {}
    for target in ready_targets:
        unit = SCORE_DEFINITIONS[target]["unit"]
        scaler = fit_robust_target_scaler(target, task_records[target], unit)
        task_records[target][REGRESSION_TARGET_COLUMN] = scaler.transform(
            task_records[target]["raw_value"]
        )
        task_records[target].to_csv(
            os.path.join(args.output_dir, "task_records", f"{target}.csv"),
            index=False,
        )
        target_scalers[target] = scaler
        print(
            f"[target-scaler] task={target} train_only=True "
            f"median={scaler.median:.6g} q1={scaler.q1:.6g} "
            f"q3={scaler.q3:.6g} iqr={scaler.iqr:.6g}",
            flush=True,
        )
    scaler_path = os.path.join(args.output_dir, "target_scalers.json")
    write_target_scalers(target_scalers, scaler_path)
    with open(
        os.path.join(args.output_dir, "split_assignment_manifest.json"),
        encoding="utf-8",
    ) as handle:
        split_assignment_manifest = json.load(handle)
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
        "experiment": (
            f"exp2_raw_video_{args.frame_policy}_head32_regression_balanced_split"
        ),
        "result_variant": args.frame_policy,
        "source_dir": os.path.abspath(args.source_dir),
        "source_data_quality_report": source_quality,
        "architectures": list(manifest_architectures),
        "targets": list(ready_targets),
        "seed": args.seed,
        "comparison_reference": (
            {
                "experiment": "exp2_face_history_head32_regression",
                "output_dir": os.path.abspath(args.reference_output_dir),
                "controlled_difference": (
                    "this experiment omits the prior-lab history input and history "
                    "encoder; image input, labels, splits, frame offsets, model "
                    "backbones, head width, optimization, and job seeds are unchanged"
                ),
                "exact_validation": (
                    "hospital_id, video_id, mirror, lab_patient_id, binary_label, "
                    "source_sample_id, split, and raw_value"
                ),
            }
            if args.reference_output_dir
            else None
        ),
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
            "reference_task_record_sha256": (
                {
                    target: _sha256_file(
                        os.path.join(
                            args.reference_output_dir,
                            "task_records",
                            f"{target}.csv",
                        )
                    )
                    for target in ready_targets
                }
                if args.reference_output_dir
                else None
            ),
            "split_distribution_audit_sha256": _sha256_file(
                os.path.join(args.output_dir, "split_distribution_audit.csv")
            ),
            "split_distribution_pairwise_sha256": _sha256_file(
                os.path.join(args.output_dir, "split_distribution_pairwise.csv")
            ),
            "split_assignment_manifest_sha256": _sha256_file(
                os.path.join(args.output_dir, "split_assignment_manifest.json")
            ),
            "target_scalers_sha256": _sha256_file(scaler_path),
        },
        "model_head": {
            "type": "Linear-LayerNorm-SiLU-Dropout-Linear",
            "hidden_features": HEAD_HIDDEN_FEATURES,
            "dropout": 0.25,
            "output_activation": None,
        },
        "regression_target": {
            "name": "raw_lab_value",
            "model_target_column": REGRESSION_TARGET_COLUMN,
            "transform": REGRESSION_TARGET_TRANSFORM,
            "formula": "(raw_value - train_median) / train_IQR",
            "inverse_formula": "raw_value = model_output * train_IQR + train_median",
            "clipping": None,
            "scalers": {
                target: scaler.to_dict() for target, scaler in target_scalers.items()
            },
            "scaler_fit_scope": "training videos only, independently per target",
            "reported_metrics": "inverse-transformed original laboratory units",
            "clinical_thresholds_for_secondary_sign_metrics": SCORE_DEFINITIONS,
            "duplicate_event_policy": (
                "one nearest in-window lab measurement per raw video and target"
            ),
            "analyte_source_policies": source_quality.get(
                "analyte_source_policies", {}
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
                "exact patient-disjoint 60/20/20 assignment reused from the "
                "face-plus-history reference experiment"
                if args.reference_output_dir
                else (
                    "patient-disjoint 60/20/20 class-stratified candidate search; "
                    "selected by raw-value and abnormal-score distribution"
                )
            ),
            "evaluation_unit": (
                (
                    f"video; mean robust-scaled raw prediction over {FRAMES_PER_VIDEO} "
                    "selected source frames, then inverse transform to raw units"
                )
                if args.frame_policy == "20frame"
                else (
                    "video; mean robust-scaled raw prediction over all indexed "
                    "frames, then inverse transform to raw units"
                )
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
            "execution_mode": (
                "add_targets" if args.add_targets else "full_output_overwrite"
                if args.overwrite else "new_run"
            ),
            "trained_targets_this_invocation": list(requested_targets),
            "preserved_targets": [
                target for target in ready_targets if target not in requested_targets
            ],
            "stage_1": (
                f"frozen encoder, lr={HEAD_LEARNING_RATE:.8g}"
            ),
            "stage_2": (
                f"all parameters unfrozen, lr={FINETUNE_LEARNING_RATE:.8g}"
            ),
            "objective": "unweighted SmoothL1 on train-only robust-scaled raw values",
            "smooth_l1_beta": SMOOTH_L1_BETA,
            "loss_weighting": "none",
            "head_max_epochs": args.head_epochs,
            "finetune_max_epochs": args.finetune_epochs,
            "scheduler": "dynamic process queue with one persistent training slot per GPU",
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
            "final_checkpoint_validation": (
                "reload model.pt on CPU and validate target/scaler/state"
            ),
        },
        "pretrained_weights": weight_manifest,
    }
    with open(
        os.path.join(args.output_dir, "experiment_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(experiment_manifest, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, "target_definition.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "target": "raw laboratory value",
                "training_transform": REGRESSION_TARGET_TRANSFORM,
                "scaler_file": os.path.abspath(scaler_path),
                "scaler_fit_scope": "train split only",
                "loss_space": "robust-scaled raw value",
                "metric_space": "inverse-transformed raw value",
                "clinical_thresholds_for_secondary_sign_metrics": SCORE_DEFINITIONS,
                "event_policy": (
                    "retain one closest lab measurement within 24 hours for each "
                    "raw video and target"
                ),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

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
        for target in requested_targets:
            jobs.append({
                "architecture": architecture,
                "target": target,
                "records_path": os.path.join(
                    args.output_dir, "task_records", f"{target}.csv"
                ),
                "target_scaler": target_scalers[target],
                "weights_dir": args.weights_dir,
                "run_dir": os.path.join(args.output_dir, "runs", architecture, target),
                "seed": args.seed,
                "head_epochs": args.head_epochs,
                "finetune_epochs": args.finetune_epochs,
                "head_patience": args.head_patience,
                "finetune_patience": args.finetune_patience,
                "max_batches": args.max_batches,
            })
    replacement_keys = {(job["architecture"], job["target"]) for job in jobs}
    preserved_run_rows = []
    preserved_metric_frames = []
    if args.add_targets:
        old_run_index = pd.read_csv(run_index_path)
        preserve_mask = ~old_run_index.apply(
            lambda row: (str(row["architecture"]), str(row["target"]))
            in replacement_keys,
            axis=1,
        )
        preserved_run_rows = old_run_index.loc[preserve_mask].to_dict("records")
        for row in preserved_run_rows:
            if str(row.get("status")) == "ok":
                row.update(
                    _validate_saved_checkpoint(
                        str(row["run_dir"]),
                        str(row["architecture"]),
                        str(row["target"]),
                    )
                )
        metrics_path = os.path.join(args.output_dir, "metrics_all.csv")
        if os.path.isfile(metrics_path):
            old_metrics = pd.read_csv(metrics_path)
            metric_preserve_mask = ~old_metrics.apply(
                lambda row: (str(row["architecture"]), str(row["target"]))
                in replacement_keys,
                axis=1,
            )
            preserved = old_metrics.loc[metric_preserve_mask].copy()
            if not preserved.empty:
                preserved_metric_frames.append(preserved)
        for job in jobs:
            shutil.rmtree(job["run_dir"], ignore_errors=True)
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
    gpu_ids = [slot % available_gpus for slot in range(worker_count)]
    print(
        f"Dynamic scheduler: jobs={len(jobs)} workers={worker_count} "
        f"available_gpus={available_gpus} assignments="
        f"{','.join(f'cuda:{gpu}' for gpu in gpu_ids)}",
        flush=True,
    )

    run_rows = preserved_run_rows
    metric_frames = preserved_metric_frames
    failure_rows = []
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    index_path = os.path.join(args.index_dir, "frame_offsets.npz")
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_worker_init,
            initargs=(index_path, gpu_queue),
        ) as executor:
            futures = {executor.submit(_worker_train, job): job for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                architecture, target, run_dir = (
                    job["architecture"], job["target"], job["run_dir"]
                )
                try:
                    result = future.result()
                    checkpoint_metadata = _validate_saved_checkpoint(
                        run_dir, architecture, target
                    )
                    metric_frames.append(result["metrics"])
                    status, reason, job_seed = "ok", "", result["job_seed"]
                except Exception as exc:
                    status, reason, job_seed = "failed", str(exc), np.nan
                    checkpoint_metadata = {
                        "model_pt_bytes": np.nan,
                        "model_pt_sha256": "",
                        "selected_stage": "",
                    }
                    os.makedirs(run_dir, exist_ok=True)
                    with open(
                        os.path.join(run_dir, "error.txt"), "w", encoding="utf-8"
                    ) as handle:
                        handle.write(traceback.format_exc())
                    failure_rows.append({
                        "architecture": architecture, "target": target, "error": str(exc)
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
                    **checkpoint_metadata,
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
                    f"[scheduler] {completed}/{len(jobs)} arch={architecture} "
                    f"task={target} status={status}", flush=True
                )
    finally:
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
