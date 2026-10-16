"""Continuous eight-GPU queue: no barrier between variants, no implicit training."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import fcntl
import gc
import json
import multiprocessing as mp
import sys

import torch

from study.common.run_video_loss_12h import sha256
from study.exp2_face_rgb_ir_head32_regression.index import load_indexes
from study.exp2_sota_ablation.audit import validate_sources
from study.exp2_sota_ablation.run import TrainingTee
from . import config
from .splits import prepare

INDEX = IR_INDEX = DEVICE = None


def initialize(queue):
    global INDEX, IR_INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX, IR_INDEX = load_indexes(config.parent.reference.RGB_INDEX, config.parent.reference.IR_INDEX)


def complete(root, contract, verify=True):
    marker = root/"complete.json"
    if not marker.exists():
        return False
    state = json.loads(marker.read_text())
    return (state["contract"] == contract and all((root/name).is_file() and (not verify or sha256(root/name) == digest)
                                                 for name, digest in state["files"].items()))


def worker(job):
    from .train import train_job
    name, target, fold = job["variant"], job["target"], job["fold"]
    root = config.OUTPUT/name/target/f"fold{fold}"
    root.mkdir(parents=True, exist_ok=True)
    for phase in ("selection", "refit"):
        file = f"{target}_fold{fold}_{phase}.csv"
        if sha256(config.SPLITS/file) != job["split_files"][file]:
            raise RuntimeError("Fold records changed before worker start")
    try:
        with (root/"train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(TrainingTee(sys.stdout, log)):
            train_job(name, target, fold, INDEX, IR_INDEX, DEVICE, job["contract"], job["split_contract"])
        return name, target, fold
    finally:
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


def code_hashes():
    folders = [config.HERE, config.parent.HERE]
    files = [path for folder in folders for path in sorted(folder.glob("*.py")) if not path.name.startswith("test_")]
    files.append(config.HERE.parent/"common/ir_compile.py")
    return {str(path.relative_to(config.HERE.parents[1])): sha256(path) for path in files}


def plan():
    return {"training_started": False, "outer_folds": 5, "train_test_patients": "80/20",
        "checkpoint_policy": "inner 80/20 of outer train selects budgets; fresh full-80% fixed-budget refit",
        "old_rgb_supervised_checkpoints_allowed": False, "test_selection": False,
        "queue": "continuous next job as each GPU finishes; no per-variant barrier",
        "final_models": len(config.VARIANTS)*len(config.TARGETS)*5,
        "variants": {name: {"based_on": config.PARENTS.get(name), "settings": config.settings(name)} for name in config.VARIANTS}}


def run_queue(jobs, executor, callback, workers):
    """Keep only one submitted job per worker; stop feeding after any failure."""
    remaining, active = iter(jobs), {}

    def submit_one():
        job = next(remaining, None)
        if job is not None:
            active[executor.submit(worker, job)] = job

    for _ in range(workers):
        submit_one()
    while active:
        done, _ = wait(active, return_when=FIRST_COMPLETED)
        finished = []
        # Resolve failures before submitting anything new from this completion set.
        for future in done:
            job = active.pop(future)
            future.result()
            finished.append(job)
        # Refill before plotting so idle GPUs do not wait for matplotlib/bootstrap.
        for _ in finished:
            submit_one()
        for job in finished:
            callback(job)


def execute(variants, devices, provenance, split_manifest):
    split_contract = sha256(config.SPLITS/"manifest.json")
    suite = {"schema": 1, "protocols": {name: config.settings(name) for name in config.VARIANTS},
        "sources": provenance, "split_contract": split_contract, "code": code_hashes(),
        "selection_policy": plan()["checkpoint_policy"], "queue": plan()["queue"]}
    protocol = config.OUTPUT/"protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != suite:
        raise RuntimeError("Source/configuration/code changed; refusing mixed five-fold results")
    protocol.write_text(json.dumps(suite, indent=2)+"\n")
    contract = sha256(protocol)
    jobs = [{"variant": name, "target": target, "fold": fold, "contract": contract,
             "split_contract": split_contract, "split_files": split_manifest["files"]}
            for name in variants for fold in range(5) for target in config.TARGETS
            if not complete(config.OUTPUT/name/target/f"fold{fold}", contract)]
    from .plots import update

    plot_future = None
    plot_pool = ThreadPoolExecutor(max_workers=1)

    def callback(job):
        nonlocal plot_future
        progress = {"variant": job["variant"], "target": job["target"], "fold": job["fold"], "status": "complete"}
        with (config.OUTPUT/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(progress)+"\n")
        # One coalescing CPU plot worker; GPU dispatch never waits for matplotlib.
        if plot_future is None or plot_future.done():
            if plot_future is not None:
                plot_future.result()
            plot_future = plot_pool.submit(update, contract)

    try:
        if jobs:
            workers = min(len(devices), len(jobs))
            context = mp.get_context("spawn")
            with context.Manager() as manager:
                queue = manager.Queue()
                for gpu in devices[:workers]:
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                        initializer=initialize, initargs=(queue,)) as executor:
                    run_queue(jobs, executor, callback, workers)
    finally:
        plot_pool.shutdown(wait=True)
    if plot_future is not None:
        plot_future.result()
    update(contract)
    (config.OUTPUT/"REQUEST_COMPLETE.json").write_text(json.dumps({"variants": variants, "test_selection": False})+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="CPU source audit and balanced split preparation only")
    parser.add_argument("--run", action="store_true", help="Explicit opt-in training")
    parser.add_argument("--variants", nargs="+", choices=tuple(config.VARIANTS), default=list(config.VARIANTS))
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.prepare and args.run:
        raise ValueError("Choose preparation or training")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Duplicate protocols")
    if not args.prepare and not args.run:
        print(json.dumps(plan(), indent=2))
        return
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT/".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, _, provenance = validate_sources()
        # The old warm-start files are audit inputs only, never model inputs here.
        provenance = {key: value for key, value in provenance.items() if key not in ("warmstarts", "baseline_checkpoints_sha256")}
        manifest = prepare()
        from .evidence import write_evidence
        write_evidence()
        from .plots import plot_distribution
        plot_distribution()
        if args.prepare:
            (config.OUTPUT/"design_plan.json").write_text(json.dumps(plan(), indent=2)+"\n")
            print(f"[prepared] {config.SPLITS}; no training launched", flush=True)
            return
        devices = [int(item) for item in args.devices.split(",")]
        if (not devices or len(set(devices)) != len(devices) or not torch.cuda.is_available()
                or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in devices)):
            raise ValueError("Distinct valid CUDA devices required")
        execute(args.variants, devices, provenance, manifest)


if __name__ == "__main__":
    main()
