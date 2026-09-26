"""Run the three face-only regression ablations sequentially on four GPUs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import traceback

import numpy as np
import pandas as pd
import torch

from .config import (
    DIRECT_LEARNING_RATE, DIRECT_PATIENCE, FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS, HEAD_PATIENCE, MIN_LEARNING_RATE, REFERENCE_INDEX_DIR, SEED, TARGETS,
    VIEW_NAMES, WEIGHTS_DIR, WEIGHT_DECAY,
)
from .frame_index import FrameOffsetIndex
from .models import WEIGHT_FILES
from .plot_results import main as plot_results
from .scaling import RobustTargetScaler
from .train import train_task


EXP_DIR = Path(__file__).resolve().parent
BASE_DIR = EXP_DIR / "outputs/20frame"
ABLATION_DIR = EXP_DIR / "outputs/ablations"
VARIANTS = (
    ("one_stage_full", "efficientnet_b0", "one_stage_full"),
    ("two_stage_tail30", "efficientnet_b0", "two_stage_tail30"),
    ("shufflenet_two_stage", "shufflenet_v2_x1_0", "two_stage_full"),
)
_FRAME_INDEX = None
_GPU_ID = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _job_seed(target):
    token = f"{SEED}:efficientnet_b0:{target}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return (SEED + offset) % (2**31 - 1)


def _prepare_sources():
    baseline = pd.read_csv(BASE_DIR / "run_index.csv")
    expected = {("efficientnet_b0", target) for target in TARGETS}
    actual = set(baseline[["architecture", "target"]].itertuples(index=False, name=None))
    if actual != expected or not baseline.status.eq("ok").all():
        raise RuntimeError("The eight EfficientNet-B0 baseline jobs must be complete")
    for row in baseline.itertuples():
        if int(row.job_seed) != _job_seed(row.target):
            raise AssertionError(f"Baseline job seed mismatch: {row.target}")
    with open(BASE_DIR / "target_scalers.json", encoding="utf-8") as handle:
        scalers = json.load(handle)["targets"]
    index_path = Path(REFERENCE_INDEX_DIR) / "frame_offsets.npz"
    index = FrameOffsetIndex.load(index_path)
    records_paths = {}
    for target in TARGETS:
        path = BASE_DIR / "task_records" / f"{target}.csv"
        records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
        if set(records.split) != {"train", "val", "test"}:
            raise AssertionError(f"Missing split for {target}")
        if records.video_id.duplicated().any():
            raise AssertionError(f"Duplicate video for {target}")
        scaler = RobustTargetScaler(**scalers[target])
        expected_scaled = scaler.transform(records.raw_value.to_numpy(float))
        if not np.allclose(expected_scaled, records.robust_scaled_raw_value, rtol=0, atol=1e-10):
            raise AssertionError(f"Saved scaler does not match labels: {target}")
        if any(index.frame_range(video_id)[1] - index.frame_range(video_id)[0] != 20
               for video_id in records.video_id):
            raise AssertionError(f"Non-20-frame video for {target}")
        patients = [set(records.loc[records.split.eq(split), "hospital_id"])
                    for split in ("train", "val", "test")]
        if any(patients[i] & patients[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise AssertionError(f"Patient leakage across splits for {target}")
        records_paths[target] = path
    source_hashes = {
        "run_index": _sha256(BASE_DIR / "run_index.csv"),
        "target_scalers": _sha256(BASE_DIR / "target_scalers.json"),
        "frame_index": _sha256(index_path),
        "task_records": {target: _sha256(path) for target, path in records_paths.items()},
    }
    print(f"[source-validated] targets={len(TARGETS)} views={len(VIEW_NAMES)} "
          f"frame_policy=20frame indexed_frames={len(index.starts)}", flush=True)
    return records_paths, scalers, index_path, source_hashes


def _init_worker(index_path, queue):
    global _FRAME_INDEX, _GPU_ID
    _GPU_ID = int(queue.get())
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(1)
    _FRAME_INDEX = FrameOffsetIndex.load(index_path)
    print(f"[worker-ready] pid={os.getpid()} gpu=cuda:{_GPU_ID}", flush=True)


def _run_job(job):
    seed = _job_seed(job["target"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str, "video_id": str})
    metrics = train_task(
        architecture=job["architecture"], target=job["target"],
        frame_index=_FRAME_INDEX, records=records,
        target_scaler=RobustTargetScaler(**job["scaler"]),
        weights_dir=WEIGHTS_DIR, run_dir=job["run_dir"],
        head_epochs=job["stage_config"]["head_max_epochs"],
        finetune_epochs=job["stage_config"]["finetune_max_epochs"],
        head_patience=job["stage_config"]["head_patience"],
        finetune_patience=job["stage_config"]["finetune_patience"],
        head_learning_rate=job["stage_config"]["head_learning_rate"],
        finetune_learning_rate=job["stage_config"]["finetune_learning_rate"],
        head_min_learning_rate=job["stage_config"]["head_min_learning_rate"],
        finetune_min_learning_rate=job["stage_config"]["finetune_min_learning_rate"],
        weight_decay=job["weight_decay"],
        freeze_batchnorm_stats=job["freeze_batchnorm_stats"],
        training_protocol=job["protocol"],
        train_batch_policy=job.get("train_batch_policy", "chunked"),
    )
    return metrics, seed


def _run_variant(name, architecture, protocol, records_paths, scalers,
                 index_path, source_hashes, train_batch_policy="chunked",
                 stage_config=None, weight_decay=WEIGHT_DECAY,
                 freeze_batchnorm_stats=False):
    output_dir = ABLATION_DIR / name
    if output_dir.exists():
        raise FileExistsError(f"Ablation output already exists: {output_dir}")
    (output_dir / "runs").mkdir(parents=True)
    weight_path = Path(WEIGHTS_DIR) / WEIGHT_FILES[architecture]
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    schedule = {
        "head_learning_rate": HEAD_LEARNING_RATE,
        "finetune_learning_rate": FINETUNE_LEARNING_RATE,
        "head_min_learning_rate": MIN_LEARNING_RATE,
        "finetune_min_learning_rate": MIN_LEARNING_RATE,
        "head_max_epochs": HEAD_MAX_EPOCHS,
        "finetune_max_epochs": FINETUNE_MAX_EPOCHS,
        "head_patience": HEAD_PATIENCE,
        "finetune_patience": FINETUNE_PATIENCE,
    }
    if stage_config:
        unknown = set(stage_config) - set(schedule)
        if unknown:
            raise ValueError(f"Unknown stage settings: {sorted(unknown)}")
        schedule.update(stage_config)
    manifest = {
        "schema_version": 1,
        "variant": name,
        "architecture": architecture,
        "training_protocol": protocol,
        "train_batch_policy": train_batch_policy,
        "weight_decay": weight_decay,
        "batchnorm_running_stats_frozen": freeze_batchnorm_stats,
        "targets": list(TARGETS),
        "baseline_output": str(BASE_DIR),
        "baseline_source_sha256": source_hashes,
        "job_seeds": {target: _job_seed(target) for target in TARGETS},
        "frame_policy": "20 non-adjacent source frames per video; saved shared index",
        "training_views": list(VIEW_NAMES),
        "head_hidden_features": 32,
        "head_learning_rate": schedule["head_learning_rate"] if protocol != "one_stage_full" else None,
        "finetune_learning_rate": schedule["finetune_learning_rate"] if protocol != "one_stage_full" else None,
        "head_min_learning_rate": schedule["head_min_learning_rate"] if protocol != "one_stage_full" else None,
        "finetune_min_learning_rate": schedule["finetune_min_learning_rate"] if protocol != "one_stage_full" else None,
        "direct_learning_rate": DIRECT_LEARNING_RATE if protocol == "one_stage_full" else None,
        "head_max_epochs": schedule["head_max_epochs"] if protocol != "one_stage_full" else None,
        "finetune_max_epochs": schedule["finetune_max_epochs"] if protocol != "one_stage_full" else None,
        "direct_max_epochs": schedule["head_max_epochs"] + schedule["finetune_max_epochs"] if protocol == "one_stage_full" else None,
        "head_patience": schedule["head_patience"] if protocol != "one_stage_full" else None,
        "finetune_patience": schedule["finetune_patience"] if protocol != "one_stage_full" else None,
        "direct_patience": DIRECT_PATIENCE if protocol == "one_stage_full" else None,
        "partial_unfreeze": "EfficientNet features[7:9] plus head; features[0:7] frozen with BN eval"
        if protocol == "two_stage_tail30" else None,
        "pretrained_weight_sha256": _sha256(weight_path),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    jobs = [
        {
            "architecture": architecture, "target": target, "protocol": protocol,
            "train_batch_policy": train_batch_policy,
            "weight_decay": weight_decay,
            "freeze_batchnorm_stats": freeze_batchnorm_stats,
            "stage_config": schedule,
            "records_path": str(records_paths[target]), "scaler": scalers[target],
            "run_dir": str(output_dir / "runs" / architecture / target),
        }
        for target in TARGETS
    ]
    context = mp.get_context("spawn")
    manager = context.Manager()
    queue = manager.Queue()
    worker_count = min(torch.cuda.device_count(), len(jobs))
    if worker_count < 1:
        raise RuntimeError("CUDA is required for the ablation experiment")
    for gpu_id in range(worker_count):
        queue.put(gpu_id)
    print(f"[variant-start] name={name} jobs={len(jobs)} gpus={worker_count}", flush=True)
    rows, metrics_frames, failed = [], [], []
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context,
            initializer=_init_worker, initargs=(str(index_path), queue),
        ) as executor:
            futures = {executor.submit(_run_job, job): job for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                run_dir = Path(job["run_dir"])
                try:
                    metrics, seed = future.result()
                    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
                    if (checkpoint.get("architecture") != architecture
                            or checkpoint.get("target") != job["target"]
                            or checkpoint.get("training_protocol") != protocol
                            or checkpoint.get("train_batch_policy") != train_batch_policy
                            or checkpoint.get("weight_decay") != weight_decay
                            or checkpoint.get("batchnorm_running_stats_frozen")
                            != freeze_batchnorm_stats):
                        raise AssertionError(f"Wrong checkpoint identity: {run_dir}")
                    metrics_frames.append(metrics)
                    status, reason = "ok", ""
                except Exception as exc:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    status, reason, seed = "failed", str(exc), np.nan
                    failed.append(job["target"])
                    print(f"[job-failed] variant={name} task={job['target']} reason={exc}", flush=True)
                rows.append({
                    "architecture": architecture, "target": job["target"],
                    "status": status, "reason": reason, "job_seed": seed,
                    "run_dir": str(run_dir),
                })
                pd.DataFrame(rows).to_csv(output_dir / "run_index.csv", index=False)
                if metrics_frames:
                    pd.concat(metrics_frames, ignore_index=True).to_csv(
                        output_dir / "metrics_all.csv", index=False
                    )
                print(f"[scheduler] variant={name} {completed}/{len(jobs)} "
                      f"target={job['target']} status={status}", flush=True)
    finally:
        manager.shutdown()
    if failed:
        raise RuntimeError(f"Ablation {name} failed for: {', '.join(failed)}")
    histories = [pd.read_csv(path) for path in sorted((output_dir / "runs").glob("*/*/history.csv"))]
    pd.concat(histories, ignore_index=True).to_csv(output_dir / "history_all.csv", index=False)
    plot_results(output_dir)
    (output_dir / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[variant-complete] name={name} output={output_dir}", flush=True)


def main():
    records_paths, scalers, index_path, source_hashes = _prepare_sources()
    for name, architecture, protocol in VARIANTS:
        _run_variant(name, architecture, protocol, records_paths, scalers,
                     index_path, source_hashes)
    from .plot_ablation_comparison import plot_comparison
    plot_comparison(BASE_DIR, ABLATION_DIR, VARIANTS)
    print("[all-ablations-complete]", flush=True)


if __name__ == "__main__":
    main()
