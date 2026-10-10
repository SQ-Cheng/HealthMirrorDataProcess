"""Extract native color data, fit all 120 models, and generate result figures."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch

from study.common.run_video_loss_12h import Tee
from study.exp6_face_pair_lab_delta.train import _metrics
from .extract import prepare
from .preview_rois import HERE, sha256
from .roi import NAMES
from .train import train_job


DEVICE = "cpu"


def init_gpu(queue):
    global DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = f"cuda:{gpu}"
    torch.set_num_threads(1)


def worker(job):
    torch.set_num_threads(1)
    output = HERE / f"outputs/models/{job['kind']}/{job['roi_mode']}/{job['target']}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        return train_job(job, DEVICE if job["kind"] == "mlp" else "cpu")


def bootstrap(prediction, seed):
    rng = np.random.default_rng(seed)
    groups = [group.index.to_numpy() for _, group in prediction.reset_index(drop=True).groupby("hospital_id")]
    actual, predicted = prediction.y_true.to_numpy(float), prediction.y_pred.to_numpy(float)
    values = []
    for _ in range(1000):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        metric = _metrics(actual[ids], predicted[ids])
        values.append([metric["mae"], metric["pearson_r"], metric["r2"]])
    intervals = np.nanquantile(values, [.025, .975], axis=0)
    return {f"{metric}_{bound}": intervals[row, col] for col, metric in enumerate(("mae", "pearson_r", "r2"))
            for row, bound in enumerate(("ci_low", "ci_high"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    args = parser.parse_args()
    output = HERE / "outputs"
    output.mkdir(exist_ok=True)
    protocol = json.loads((HERE / "protocol.json").read_text())
    modes = {"_".join(names) if len(names) == 1 else "both_cheeks" if len(names) == 2 else "all_rois":
             [NAMES.index(name) for name in names] for names in protocol["roi_configurations"]}
    with (output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.reuse_prepared:
            prepare(protocol)
        if args.prepare_only:
            return
        contract = sha256(output / "tables/preparation_manifest.json")
        jobs, rows = {"mlp": [], "ridge": []}, []
        for target in protocol["targets"]:
            for mode, indices in modes.items():
                for kind in jobs:
                    run = output / f"models/{kind}/{mode}/{target}"
                    marker = run / "job_complete.json"
                    if marker.is_file() and json.loads(marker.read_text()) == {"contract": contract} and (run / "metrics.csv").is_file():
                        rows.append({"target": target, "roi_mode": mode, "kind": kind, "status": "ok"})
                    else:
                        jobs[kind].append({"target": target, "roi_mode": mode, "roi_indices": indices, "kind": kind, "contract": contract})
        ctx = mp.get_context("spawn")
        workers = min(4, torch.cuda.device_count())
        if jobs["mlp"] and not workers:
            raise RuntimeError("CUDA required for MLP queue")
        print(f"[training-start] Exp1 MLP={len(jobs['mlp'])} ridge={len(jobs['ridge'])} GPU={workers} CPU_ridge=2", flush=True)
        with ctx.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers):
                queue.put(gpu)
            with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=init_gpu, initargs=(queue,)) as gpu_pool, \
                 ProcessPoolExecutor(max_workers=2, mp_context=ctx) as cpu_pool:
                futures = {pool.submit(worker, job): job for kind, pool in (("mlp", gpu_pool), ("ridge", cpu_pool)) for job in jobs[kind]}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        row = future.result()
                    except Exception:
                        row = {"target": job["target"], "roi_mode": job["roi_mode"], "kind": job["kind"], "status": "failed", "error": traceback.format_exc()}
                        print(row["error"], flush=True)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "tables/run_index.csv", index=False)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Exp1 jobs failed; see run_index.csv")
        metrics, ci = [], []
        for target in protocol["targets"]:
            for mode in modes:
                reference = None
                for kind in jobs:
                    run = output / f"models/{kind}/{mode}/{target}"
                    metrics.append(pd.read_csv(run / "metrics.csv"))
                    predictions = pd.read_csv(run / "predictions.csv", dtype={"hospital_id": str, "video_id": str}).query("split == 'test'").reset_index(drop=True)
                    identity = predictions[["hospital_id", "video_id", "y_true"]]
                    if reference is not None:
                        pd.testing.assert_frame_equal(reference, identity)
                    reference = identity
                    ci.append({"target": target, "roi_mode": mode, "kind": kind, **bootstrap(predictions, 20261010)})
        pd.concat(metrics, ignore_index=True).to_csv(output / "tables/metrics_all.csv", index=False)
        pd.DataFrame(ci).to_csv(output / "tables/test_patient_bootstrap.csv", index=False)
        from .plots import plot_results
        plot_results(protocol, modes)
        (output / "COMPLETE").write_text("120 native41 ROI regressors, patient-group ridge CV, bootstrap and figures completed\n")
        print("[queue-complete] Exp1 native ROI color regression", flush=True)


if __name__ == "__main__":
    main()
