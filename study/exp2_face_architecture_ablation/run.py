"""Wait for the matched view-loss control, then schedule 48 lightweight models."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import traceback
import sys
import time

import pandas as pd
import torch

from study.common.run_distinct_lab_views_12h import prepared_records
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex

from . import config
from .features import ensure_cache
from .models import build_model


INDEX = DEVICE = None


def predecessor_complete():
    path = config.PREDECESSOR / ".queue.lock"
    if not path.is_file():
        raise RuntimeError("The preceding distinct-lab queue has not been registered")
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (config.PREDECESSOR / "COMPLETE").is_file():
            raise RuntimeError("The preceding queue stopped before completion")
    if not all((root / "COMPLETE").is_file() for root in config.BASELINES.values()):
        raise RuntimeError("A matched EfficientNet control is incomplete")
    return True


def init_worker(queue):
    global INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX = FrameOffsetIndex.load(config.INDEX_PATH)


def worker(job):
    from .train import train_task
    run = config.OUTPUT_DIR / job["family"] / job["architecture"] / "runs" / job["target"]
    run.mkdir(parents=True, exist_ok=True)
    with (run / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(config.reference.Tee(sys.stdout, log)):
        train_task(job, INDEX, DEVICE)
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != job["architecture"] or checkpoint["loss_unit"] != "video_view":
        raise AssertionError("Wrong saved architecture/objective")
    (run / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}))
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return {"architecture": job["architecture"], "family": job["family"], "target": job["target"], "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    _, scalers, hashes, counts = config.reference.preflight()
    texts, audit = prepared_records()
    source_hashes = {target: hashlib.sha256(text.encode()).hexdigest() for target, text in texts.items()}
    parameters = {architecture: sum(p.numel() for p in build_model(architecture).parameters())
                  for architecture in config.ARCHITECTURES}
    if args.check_only:
        print(json.dumps(parameters, indent=2))
        print(audit.to_string(index=False))
        return
    output = config.OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {
            "architectures": list(config.ARCHITECTURES), "families": ["classification", "regression"],
            "targets": list(config.TARGETS), "parameters": parameters,
            "matching_hours": 12, "source_resolution": 224, "frames_per_view": 20,
            "training_views": list(config.VIEWS), "batch_unique_measurements": 12,
            "frame_batch_size": 240, "loss_unit": "video_view", "source_records_sha256": source_hashes,
            "frame_index_sha256": config.reference.sha256(config.INDEX_PATH),
            "scalers_sha256": config.reference.sha256(config.SOURCE / "target_scalers.json"),
            "predecessor": str(config.PREDECESSOR), "baselines": {k: str(v) for k, v in config.BASELINES.items()},
            "epochs": config.MAX_EPOCHS, "patience": config.PATIENCE,
            "learning_rates": config.LEARNING_RATES, "minimum_learning_rates": config.MIN_LEARNING_RATES,
            "weight_decay": config.WEIGHT_DECAY, "dropout": config.DROPOUT,
            "optimizer": "AdamW", "scheduler": "cosine; no warmup", "compile": False,
            "initialization": "from scratch; all parameters trained jointly",
            "feature_scaling": "mean/std fitted to train frames and views only; std floor 1e-4",
            "selection": "classification val bACC; regression raw-unit val MAE",
            "comparability": "same data/split/views/loss/batch; architectures, pretraining, and optimization differ",
            "job_seeds": {family: {target: config.reference.job_seed(family, target) for target in config.TARGETS}
                          for family in ("classification", "regression")},
        }
        path = output / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Registered architecture experiment contract changed")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        audit.to_csv(output / "batch_audit.csv", index=False)
        counts.to_csv(output / "cohort_counts.csv", index=False)
        while not predecessor_complete():
            print("[waiting] distinct-lab view-loss controls active; no GPU or feature extraction allocated", flush=True)
            time.sleep(60)
        if config.reference.preflight()[2] != hashes or prepared_records()[0] != texts:
            raise RuntimeError("Clinical records changed while waiting")
        records_dir = output / "source_records"
        records_dir.mkdir(exist_ok=True)
        for target, text in texts.items():
            (records_dir / f"{target}.csv").write_text(text)
            for family, root in config.BASELINES.items():
                if config.reference.sha256(root / f"task_records/{target}.csv") != source_hashes[target]:
                    raise RuntimeError(f"Baseline split/measurement IDs differ: {family}/{target}")
        ensure_cache()
        contract = config.reference.sha256(path)
        jobs, rows = [], []
        for target in config.TARGETS:
            for architecture in ("small_cnn", "color_histogram_mlp", "color_statistics_mlp"):
                for family in ("classification", "regression"):
                    run = output / family / architecture / "runs" / target
                    marker = run / "job_complete.json"
                    if marker.is_file() and json.loads(marker.read_text()) == {"contract": contract} and all(
                        (run / name).is_file() for name in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv")
                    ):
                        rows.append({"architecture": architecture, "family": family, "target": target, "status": "ok"})
                    else:
                        jobs.append({"architecture": architecture, "family": family, "target": target,
                                     "scaler": scalers[target], "contract": contract})
        workers = min(4, torch.cuda.device_count(), len(jobs))
        if jobs and workers < 1:
            raise RuntimeError("CUDA required for formal training")
        print(f"[scheduler] architecture controls jobs=48 pending={len(jobs)} gpus={workers}", flush=True)
        ctx = mp.get_context("spawn")
        if jobs:
            with ctx.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker, initargs=(queue,)) as pool:
                    futures = {pool.submit(worker, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            row = future.result()
                        except Exception:
                            row = {"architecture": job["architecture"], "family": job["family"], "target": job["target"],
                                   "status": "failed", "error": traceback.format_exc()}
                            print(row["error"], flush=True)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
                        print(f"[task-finished] {job['architecture']}/{job['family']}/{job['target']} {row['status']}", flush=True)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Architecture jobs failed; complete models can be reused")
        from .plots import plot_one, compare
        for family in ("classification", "regression"):
            for architecture in config.ARCHITECTURES:
                root = output / family / architecture
                for name in ("metrics", "history"):
                    pd.concat([pd.read_csv(root / f"runs/{target}/{name}.csv") for target in config.TARGETS], ignore_index=True).to_csv(root / f"{name}_all.csv", index=False)
                plot_one(root, family, architecture)
                (root / "COMPLETE").write_text("eight tasks and figures completed\n")
        compare()
        (output / "COMPLETE").write_text("48 model jobs and paired architecture figures completed\n")
        print("[queue-complete] architecture controls and matched comparisons", flush=True)


if __name__ == "__main__":
    main()
