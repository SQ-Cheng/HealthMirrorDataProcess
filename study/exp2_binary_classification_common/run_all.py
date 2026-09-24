"""Dynamically schedule all three binary experiments over available GPUs."""

import argparse
import multiprocessing as mp
from pathlib import Path
import shutil
import traceback

import pandas as pd
import torch

from study.exp2_lab_multimodal.config import SEED

from .engine import EXPERIMENT_DIRS, MODALITIES, TARGETS, train_task
from .plot_results import plot_all
from .prepare import prepare


def _worker(device_id, tasks, results, seed, smoke):
    while True:
        item = tasks.get()
        if item is None:
            return
        modality, target = item
        try:
            train_task(modality, target, device_id, seed, smoke=smoke)
            results.put((modality, target, "complete", ""))
        except Exception:
            results.put((modality, target, "failed", traceback.format_exc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()
    if not args.skip_prepare:
        prepare()
    if args.prepare_only:
        return
    if args.overwrite:
        for experiment_dir in EXPERIMENT_DIRS.values():
            shutil.rmtree(experiment_dir / "outputs", ignore_errors=True)
    if args.gpus:
        devices = [int(value) for value in args.gpus.split(",")]
    else:
        devices = list(range(torch.cuda.device_count()))
    if not devices:
        raise RuntimeError("No CUDA devices available")
    jobs = (
        [(modality, "total_bilirubin_high") for modality in MODALITIES]
        if args.smoke_test
        else [(modality, target) for modality in MODALITIES for target in TARGETS]
    )
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for job in jobs:
        tasks.put(job)
    for _ in devices[:len(jobs)]:
        tasks.put(None)
    workers = [
        ctx.Process(target=_worker, args=(device, tasks, results, args.seed, args.smoke_test))
        for device in devices[:len(jobs)]
    ]
    for worker in workers:
        worker.start()
    rows = [results.get() for _ in jobs]
    for worker in workers:
        worker.join()
    failures = [row for row in rows if row[2] == "failed"]
    status = pd.DataFrame(rows, columns=("modality", "target", "status", "error"))
    for modality in MODALITIES:
        output = EXPERIMENT_DIRS[modality] / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        status[status.modality.eq(modality)].to_csv(output / "run_index.csv", index=False)
    if failures:
        for modality, target, _, error in failures:
            print(f"[job-failed] {modality}/{target}\n{error}", flush=True)
        raise RuntimeError(f"{len(failures)} binary-classification jobs failed")
    if not args.smoke_test:
        plot_all()
    print(f"[all-complete] jobs={len(jobs)} smoke={args.smoke_test}", flush=True)


if __name__ == "__main__":
    main()
