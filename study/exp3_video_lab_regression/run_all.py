"""Prepare Exp3 or dynamically train eight independent R3D-18 regressors."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import traceback

import pandas as pd
import torch

from .clips import ClipIndex
from .config import INDEX_DIR, OUTPUT_DIR, SEED, TARGETS
from .plot_results import plot_results
from .prepare import prepare
from .train import train_task


_INDEX = None
_GPU_ID = None


def _worker_init(index_path, gpu_queue):
    global _INDEX, _GPU_ID
    _GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(2)
    _INDEX = ClipIndex.load(index_path)
    print(f"[worker-ready] pid={os.getpid()} gpu=cuda:{_GPU_ID} "
          f"indexed_clips={len(_INDEX.starts)}", flush=True)


def _worker_train(job):
    token = f"{SEED}:efficientnet_b0:{job['target']}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    seed = (SEED + offset) % (2**31 - 1)
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str, "video_id": str})
    values = train_task(job["target"], records, job["scaler"], _INDEX,
                        _GPU_ID, job["run_dir"], seed)
    return {"target": job["target"], "gpu": _GPU_ID, "metrics": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        print("[prepare-only] no model training started", flush=True)
        return
    if (OUTPUT_DIR / "COMPLETE").exists() or (
        (OUTPUT_DIR / "runs").exists() and any((OUTPUT_DIR / "runs").iterdir())
    ):
        raise RuntimeError("Exp3 training outputs already exist; refusing to overwrite")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Exp3 training requires CUDA")
    with open(OUTPUT_DIR / "target_scalers.json", encoding="utf-8") as handle:
        scalers = json.load(handle)["targets"]
    jobs = [{
        "target": target, "scaler": scalers[target],
        "records_path": str(OUTPUT_DIR / "task_records" / f"{target}.csv"),
        "run_dir": str(OUTPUT_DIR / "runs" / target),
    } for target in TARGETS]
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    worker_count = min(args.workers or gpu_count, gpu_count, len(jobs))
    for gpu_id in range(worker_count):
        gpu_queue.put(gpu_id)
    print(f"[scheduler] jobs={len(jobs)} workers={worker_count} "
          f"gpus={gpu_count} dynamic_queue=true", flush=True)
    rows, metrics, failures = [], [], []
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context,
        initializer=_worker_init,
        initargs=(str(INDEX_DIR / "clip_offsets.npz"), gpu_queue),
    ) as executor:
        futures = {executor.submit(_worker_train, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                metrics.append(pd.DataFrame(result["metrics"]))
                rows.append({"target": job["target"], "status": "ok",
                             "gpu": result["gpu"], "run_dir": job["run_dir"]})
                print(f"[scheduler-complete] target={job['target']} "
                      f"gpu=cuda:{result['gpu']}", flush=True)
            except Exception as exc:
                failure = {"target": job["target"], "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                rows.append(failure)
                failures.append(failure)
                print(f"[scheduler-failed] target={job['target']} "
                      f"error={failure['error']}", flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "run_index.csv", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(
            OUTPUT_DIR / "metrics_all.csv", index=False
        )
    if failures:
        (OUTPUT_DIR / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} Exp3 jobs failed")
    plot_results(OUTPUT_DIR)
    (OUTPUT_DIR / "COMPLETE").write_text("ok\n", encoding="ascii")
    print("[experiment-complete] eight tasks and figures generated", flush=True)


if __name__ == "__main__":
    main()
