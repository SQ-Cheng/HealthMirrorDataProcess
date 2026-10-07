"""Queue frozen DINOv3-S head fitting after the existing architecture controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback

import pandas as pd
import torch

from study.common.run_distinct_lab_views_12h import prepared_records
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_architecture_ablation.train import train_task

from . import config
from .backbone import ensure_source, weight_sha256
from .data import loader, build_head, no_feature_fitting
from .features import ensure_cache


INDEX = DEVICE = None


def predecessor_complete():
    path = config.PREDECESSOR / ".queue.lock"
    if not path.is_file():
        raise RuntimeError("The architecture-control predecessor is not registered")
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (config.PREDECESSOR / "COMPLETE").is_file():
            raise RuntimeError("The preceding architecture queue stopped before completion")
    return True


def init_worker(queue):
    global INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX = FrameOffsetIndex.load(config.INDEX_PATH)


def worker(job):
    root = config.HERE / "outputs" / job["family"] / config.ARCHITECTURE / "runs" / job["target"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(config.reference.Tee(sys.stdout, log)):
        train_task(job, INDEX, DEVICE, experiment_config=config, model_factory=build_head,
                   loader_factory=loader, feature_scaler=no_feature_fitting)
    checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["parameters"] != 12417 or checkpoint["loss_unit"] != "video_view":
        raise RuntimeError("Saved DINO head has the wrong parameters/objective")
    checkpoint.update(backbone_frozen=True, backbone_weight_sha256=job["weight_sha256"],
                      backbone_source_revision=config.REPO_REVISION, feature="normalized CLS token")
    torch.save(checkpoint, root / "model.pt")
    (root / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}))
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return {"architecture": config.ARCHITECTURE, "family": job["family"], "target": job["target"], "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ensure_source()
    _, scalers, hashes, counts = config.reference.preflight()
    texts, audit = prepared_records()
    prepared_hashes = {target: hashlib.sha256(text.encode()).hexdigest() for target, text in texts.items()}
    if args.check_only:
        print(audit.to_string(index=False))
        print(f"authorized_weights_present={config.WEIGHTS.is_file()}")
        if config.WEIGHTS.is_file():
            print(f"weight_sha256={weight_sha256()}")
        return
    output = config.HERE / "outputs"
    output.mkdir(exist_ok=True)
    with (output / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {
            "architecture": config.ARCHITECTURE, "repo_revision": config.REPO_REVISION,
            "weights": str(config.WEIGHTS), "required_hash_prefix": config.WEIGHT_HASH_PREFIX,
            "predecessor": str(config.PREDECESSOR), "targets": list(config.TARGETS),
            "source_records_sha256": prepared_hashes, "backbone_frozen": True,
            "backbone_parameters": 21601152, "head_parameters": 12417, "head_hidden": 32,
            "matching_hours": 12, "input_resolution": 224, "frames_per_view": 20,
            "training_views": list(config.VIEWS), "batch_distinct_measurements": 12,
            "frame_batch_size": 240, "loss_unit": "video_view", "stages": 1,
            "learning_rate": 2e-4, "minimum_lr": 1e-6, "max_epochs": 80, "patience": 12,
            "weight_decay": 1e-3, "dropout": .25, "optimizer": "AdamW", "scheduler": "cosine",
            "feature": "384-dimensional final normalized CLS token; no patch-token averaging",
            "feature_fitting": "none; official frozen final LayerNorm retained",
        }
        path = output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise RuntimeError("Registered frozen-DINO experiment plan changed")
        path.write_text(json.dumps(plan, indent=2) + "\n")
        audit.to_csv(output / "batch_audit.csv", index=False)
        counts.to_csv(output / "cohort_counts.csv", index=False)
        while True:
            previous_done = predecessor_complete()
            if not config.WEIGHTS.is_file():
                print(f"[waiting-weights] official authorized ViT-S/16 checkpoint missing: {config.WEIGHTS}; no GPU allocated", flush=True)
            elif not previous_done:
                weight_sha256()
                print("[waiting] architecture controls active; no GPU allocated", flush=True)
            else:
                break
            time.sleep(60)
        digest = weight_sha256()
        if config.reference.preflight()[2] != hashes or prepared_records()[0] != texts:
            raise RuntimeError("Clinical source changed while waiting")
        records = output / "source_records"
        records.mkdir(exist_ok=True)
        for target, text in texts.items():
            (records / f"{target}.csv").write_text(text)
            for baseline in config.BASELINES.values():
                if config.reference.sha256(baseline / f"task_records/{target}.csv") != prepared_hashes[target]:
                    raise RuntimeError("The paired EfficientNet split differs")
        manifest = {**plan, "backbone_weight_sha256": digest,
                    "frame_index_sha256": config.reference.sha256(config.INDEX_PATH)}
        (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        contract = config.reference.sha256(output / "experiment_manifest.json")
        ensure_cache()
        jobs, rows = [], []
        for target in config.TARGETS:
            for family in ("classification", "regression"):
                root = output / family / config.ARCHITECTURE / "runs" / target
                marker = root / "job_complete.json"
                if marker.exists() and json.loads(marker.read_text()) == {"contract": contract} and all(
                    (root / name).is_file() for name in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv")
                ):
                    rows.append({"architecture": config.ARCHITECTURE, "family": family, "target": target, "status": "ok"})
                else:
                    jobs.append({"architecture": config.ARCHITECTURE, "family": family, "target": target,
                                 "contract": contract, "scaler": scalers[target], "weight_sha256": digest})
        workers = min(4, torch.cuda.device_count(), len(jobs))
        if jobs and workers < 1:
            raise RuntimeError("CUDA is required")
        print(f"[scheduler] frozen DINO heads pending={len(jobs)} gpus={workers}", flush=True)
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
                            row = {"architecture": config.ARCHITECTURE, "family": job["family"], "target": job["target"],
                                   "status": "failed", "error": traceback.format_exc()}
                            print(row["error"], flush=True)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
                        print(f"[task-finished] {job['family']}/{job['target']} {row['status']}", flush=True)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Frozen-DINO head jobs failed")
        for family in ("classification", "regression"):
            root = output / family / config.ARCHITECTURE
            for name in ("metrics", "history"):
                pd.concat([pd.read_csv(root / f"runs/{target}/{name}.csv") for target in config.TARGETS], ignore_index=True).to_csv(root / f"{name}_all.csv", index=False)
        from .plots import plot_results
        plot_results()
        (output / "COMPLETE").write_text("16 frozen-DINO head models and paired figures completed\n")
        print("[queue-complete] frozen DINOv3-S regression/classification heads", flush=True)


if __name__ == "__main__":
    main()
