"""Explicit opt-in training: serial ablations, parallel analytes, incremental PNGs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import json
import multiprocessing as mp
import sys
import traceback

import pandas as pd
import torch

from study.common.run_video_loss_12h import sha256, Tee
from study.exp2_face_rgb_ir_head32_regression.index import load_indexes
from . import config
from .audit import validate_sources, write_baseline_audit


INDEX = IR_INDEX = DEVICE = None


class TrainingTee(Tee):
    @property
    def encoding(self):
        return getattr(self.stream, "encoding", None) or "utf-8"


def initialize(queue):
    global INDEX, IR_INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX, IR_INDEX = load_indexes(config.reference.RGB_INDEX, config.reference.IR_INDEX)


def worker(job):
    from .train import train_task
    target, name = job["target"], job["variant"]
    path = config.OUTPUT / name / "runs" / target
    path.mkdir(parents=True, exist_ok=True)
    source = job["provenance"]
    if sha256(config.BASELINE / f"source_records/{target}.csv") != source["records_sha256"][target]:
        raise RuntimeError("Training records changed")
    warm = source["warmstarts"][target]
    if config.VARIANTS[name]["warm_start"] and sha256(warm["path"]) != warm["sha256"]:
        raise RuntimeError("RGB initialization changed")
    try:
        with (path / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(TrainingTee(sys.stdout, log)):
            train_task(name, target, INDEX, IR_INDEX, DEVICE,
                       {"job_seeds": source["job_seeds"], "contract": job["contract"]}, warm["path"])
        return {"variant": name, "target": target, "status": "ok"}
    finally:
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


def completed_variant(name):
    path = config.OUTPUT / name
    protocol = path / "protocol.json"
    if not protocol.is_file() or not (path / "COMPLETE").is_file():
        return False
    contract = sha256(protocol)
    for target in config.TARGETS:
        run = path / "runs" / target
        marker = run / "job_complete.json"
        if (not marker.exists() or json.loads(marker.read_text()) != {"contract": contract}
                or not all((run / name).is_file() for name in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv"))):
            return False
    return True


def execute(name, provenance, devices):
    root = config.OUTPUT / name
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"variant": name, "settings": config.get_variant(name), "sources": provenance,
                "selection": "per-target minimum video-level validation raw MAE; no test selection",
                "target_policy": "exact same ten fixed baseline cohorts/patient splits; no rematching or data dropping"}
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError(f"Changed protocol in {name}; refusing to mix results")
    protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
    contract = sha256(protocol_path)
    jobs, rows = [], []
    for target in config.TARGETS:
        path = root / "runs" / target
        marker = path / "job_complete.json"
        if (marker.exists() and json.loads(marker.read_text()) == {"contract": contract}
                and all((path / field).exists() for field in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv"))):
            rows.append({"variant": name, "target": target, "status": "ok"})
        else:
            jobs.append({"variant": name, "target": target, "provenance": provenance, "contract": contract})
    (root / "COMPLETE").unlink(missing_ok=True)
    workers = min(len(devices), len(jobs))
    print(f"[variant-start] {name}: remaining={len(jobs)} GPUs={workers}", flush=True)
    if jobs:
        context = mp.get_context("spawn")
        with context.Manager() as manager:
            queue = manager.Queue()
            for device in devices[:workers]:
                queue.put(device)
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize, initargs=(queue,)) as pool:
                futures = {pool.submit(worker, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        rows.append(future.result())
                    except Exception:
                        rows.append({"variant": name, "target": job["target"], "status": "failed", "error": traceback.format_exc()})
                        print(rows[-1]["error"], flush=True)
                    pd.DataFrame(rows).to_csv(root / "run_index.csv", index=False)
    if len(rows) != len(config.TARGETS) or any(row["status"] != "ok" for row in rows):
        raise RuntimeError(f"{name} failed; subsequent ablations are not launched")
    for field in ("history", "metrics"):
        pd.concat([pd.read_csv(root / f"runs/{target}/{field}.csv") for target in config.TARGETS], ignore_index=True).to_csv(root / f"{field}_all.csv", index=False)
    from .plots import plot_one_variant
    plot_one_variant(name)
    (root / "COMPLETE").write_text(f"{len(config.TARGETS)} tasks completed, own PNGs complete\n")
    from .plots import update_comparisons
    update_comparisons()
    print(f"[variant-complete] {name}; comparison figures refreshed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Read-only CPU source/baseline audit; no training")
    parser.add_argument("--run", action="store_true", help="Explicitly authorize the full training suite")
    parser.add_argument("--variants", nargs="+", choices=tuple(config.VARIANTS), default=list(config.VARIANTS))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.audit and args.run:
        raise ValueError("Audit and training are separate operations")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Repeated variants")
    if not args.audit and not args.run:
        print(json.dumps({"training_started": False, "ordered_variants": {name: config.get_variant(name) for name in args.variants}}, indent=2))
        return
    index, _, provenance = validate_sources()
    if args.audit:
        write_baseline_audit(index, provenance)
        return
    devices = [int(item) for item in args.devices.split(",")]
    if (not devices or len(set(devices)) != len(devices) or not torch.cuda.is_available()
            or any(device < 0 or device >= torch.cuda.device_count() for device in devices)):
        raise ValueError("Valid distinct CUDA devices required")
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in args.variants:
            # Resume safely, but always refresh figures before advancing.
            execute(name, provenance, devices)
        if validate_sources()[2] != provenance:
            raise RuntimeError("Sources changed during the suite")
        (config.OUTPUT / "SUITE_COMPLETE.json").write_text(json.dumps({"variants": args.variants, "test_selection": False}, indent=2)+"\n")
    print("[suite-complete] ordered ablations and incremental PNG comparisons", flush=True)


if __name__ == "__main__":
    main()
