"""Run patient-diverse scheduling and middle-48-frame ablations sequentially."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import pandas as pd
import torch

from .clips import ClipIndex, build_or_reuse_index
from .config import (
    CLIP_FRAMES, CLIP_POSITIONS, FINETUNE_EPOCHS, FINETUNE_LR,
    FINETUNE_MIN_LR, FINETUNE_PATIENCE, HEAD_EPOCHS, HEAD_LR,
    HEAD_MIN_LR, HEAD_PATIENCE, INDEX_DIR, OUTPUT_DIR, SEED, TARGETS,
)
from .plot_ablations import plot_comparison
from .plot_results import plot_results
from .prepare import prepare
from .train import train_task


ABLATION_DIR = OUTPUT_DIR / "ablations"
MIDDLE48_INDEX_DIR = OUTPUT_DIR.parent / "cache/middle48_index"
VARIANTS = ("patient_diverse_schedule_30_40", "middle48")
BASE_SCHEDULE = {
    "head_epochs": HEAD_EPOCHS, "head_patience": HEAD_PATIENCE,
    "head_lr": HEAD_LR, "head_min_lr": HEAD_MIN_LR,
    "finetune_epochs": FINETUNE_EPOCHS,
    "finetune_patience": FINETUNE_PATIENCE,
    "finetune_lr": FINETUNE_LR, "finetune_min_lr": FINETUNE_MIN_LR,
}
DIVERSE_SCHEDULE = {
    "head_epochs": 30, "head_patience": 8,
    "head_lr": 1e-4, "head_min_lr": 1e-6,
    "finetune_epochs": 40, "finetune_patience": 8,
    "finetune_lr": 3e-6, "finetune_min_lr": 1e-7,
}
_INDEX = None
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


def _worker_init(index_path, queue):
    global _INDEX, _GPU_ID
    _GPU_ID = int(queue.get())
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(2)
    _INDEX = ClipIndex.load(index_path)
    print(f"[worker-ready] pid={os.getpid()} gpu=cuda:{_GPU_ID} "
          f"clip_frames={_INDEX.starts.shape[1]}", flush=True)


def _worker_train(job):
    records = pd.read_csv(job["records_path"],
                          dtype={"hospital_id": str, "video_id": str})
    metrics = train_task(
        job["target"], records, job["scaler"], _INDEX,
        _GPU_ID, job["run_dir"], _job_seed(job["target"]),
        batch_policy=job["batch_policy"], schedule=job["schedule"],
        variant=job["variant"],
    )
    return metrics


def _source_records():
    return {
        target: pd.read_csv(
            OUTPUT_DIR / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str},
        ) for target in TARGETS
    }


def prepare_middle48_index():
    records = _source_records()
    videos = pd.concat(
        [frame[["video_id", "mirror", "lab_patient_id"]]
         for frame in records.values()], ignore_index=True,
    )
    return build_or_reuse_index(
        videos, MIDDLE48_INDEX_DIR, clip_frames=48, positions=(0.5,)
    )


def _run_variant(name, index, index_path, source, scalers, batch_policy, schedule,
                 positions=CLIP_POSITIONS):
    output = ABLATION_DIR / name
    if output.exists():
        raise FileExistsError(f"Ablation output already exists: {output}")
    record_dir = output / "task_records"
    record_dir.mkdir(parents=True)
    summary = []
    for target, records in source.items():
        selected = records.loc[records.video_id.astype(str).isin(index.lookup)].copy()
        if set(selected.split) != {"train", "val", "test"}:
            raise RuntimeError(f"Missing split for {target}/{name}")
        selected.to_csv(record_dir / f"{target}.csv", index=False)
        for split in ("train", "val", "test"):
            before = records.loc[records.split.eq(split)]
            after = selected.loc[selected.split.eq(split)]
            summary.append({
                "target": target, "split": split,
                "baseline_videos": len(before), "usable_videos": len(after),
                "excluded_videos": len(before) - len(after),
                "patients": after.hospital_id.nunique(),
            })
    pd.DataFrame(summary).to_csv(output / "data_summary.csv", index=False)
    manifest = {
        "schema_version": 1, "variant": name,
        "base_experiment": "exp3_video_lab_regression",
        "base_manifest_sha256": _sha256(OUTPUT_DIR / "experiment_manifest.json"),
        "index_sha256": _sha256(index_path),
        "target_scalers_sha256": _sha256(OUTPUT_DIR / "target_scalers.json"),
        "clip_frames": int(index.starts.shape[1]),
        "clip_positions": list(positions),
        "batch_policy": batch_policy, "schedule": schedule,
        "targets": list(TARGETS),
        "job_seeds": {target: _job_seed(target) for target in TARGETS},
        "split_policy": "reuse baseline patient-disjoint split, filter only if no valid clip",
        "scaler_policy": "reuse baseline train-only median/IQR without refitting",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "target_scalers.json").write_bytes(
        (OUTPUT_DIR / "target_scalers.json").read_bytes()
    )
    jobs = [{
        "target": target, "records_path": str(record_dir / f"{target}.csv"),
        "scaler": scalers[target], "run_dir": str(output / "runs" / target),
        "batch_policy": batch_policy, "schedule": schedule, "variant": name,
    } for target in TARGETS]
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Ablation training requires CUDA")
    workers = min(4, gpu_count, len(jobs))
    context = mp.get_context("spawn")
    manager = context.Manager()
    queue = manager.Queue()
    for gpu_id in range(workers):
        queue.put(gpu_id)
    print(f"[variant-start] name={name} jobs={len(jobs)} gpus={workers} "
          f"clip_frames={index.starts.shape[1]}", flush=True)
    rows, metrics, failures = [], [], []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_worker_init,
        initargs=(str(index_path), queue),
    ) as executor:
        futures = {executor.submit(_worker_train, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                metrics.append(pd.DataFrame(result))
                rows.append({"target": job["target"], "status": "ok",
                             "run_dir": job["run_dir"]})
                print(f"[job-complete] variant={name} target={job['target']}", flush=True)
            except Exception as exc:
                failures.append(job["target"])
                rows.append({"target": job["target"], "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})
                print(f"[job-failed] variant={name} target={job['target']} "
                      f"error={exc}", flush=True)
            pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(
            output / "metrics_all.csv", index=False
        )
    if failures:
        raise RuntimeError(f"Failed jobs in {name}: {failures}")
    plot_results(output)
    (output / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[variant-complete] name={name}", flush=True)


def main():
    if not (OUTPUT_DIR / "COMPLETE").is_file():
        raise RuntimeError("Exp3 baseline must be complete")
    if any((ABLATION_DIR / name).exists() for name in VARIANTS):
        raise FileExistsError("One of the Exp3 ablation outputs already exists")
    baseline_index = prepare()
    source = _source_records()
    with open(OUTPUT_DIR / "target_scalers.json", encoding="utf-8") as handle:
        scalers = json.load(handle)["targets"]
    middle48_index = prepare_middle48_index()
    print(f"[middle48-index] videos={len(middle48_index.video_ids)} "
          f"clips={len(middle48_index.starts)}", flush=True)
    _run_variant(VARIANTS[0], baseline_index, INDEX_DIR / "clip_offsets.npz",
                 source, scalers, "patient_diverse", DIVERSE_SCHEDULE)
    _run_variant(VARIANTS[1], middle48_index,
                 MIDDLE48_INDEX_DIR / "clip_offsets.npz",
                 source, scalers, "random_video", BASE_SCHEDULE,
                 positions=(0.5,))
    plot_comparison(OUTPUT_DIR, tuple(ABLATION_DIR / name for name in VARIANTS))
    print("[exp3-ablations-complete]", flush=True)


if __name__ == "__main__":
    main()
