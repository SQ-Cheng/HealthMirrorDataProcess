"""Run the nine Exp6 direction-classification tasks on four GPUs."""

import argparse
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

from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex
from study.exp6_face_pair_lab_delta.config import (
    CACHE_DIR, PRETRAINED_WEIGHT_FILE, SEED, TARGETS, WEIGHTS_DIR,
)
from study.exp6_face_pair_lab_delta.run_all import (
    _load_prepared, _validate_against_baseline, _validate_records,
)

from .plot_results import plot_results
from .train import train_task


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
_FRAME_INDEX = None
_GPU_ID = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_init(index_path, gpu_queue):
    global _FRAME_INDEX, _GPU_ID
    _GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(2)
    _FRAME_INDEX = FrameOffsetIndex.load(index_path)
    print(f"[worker-ready] pid={os.getpid()} gpu=cuda:{_GPU_ID} "
          f"indexed_frames={len(_FRAME_INDEX.starts)}", flush=True)


def _worker_train(job):
    token = f"{SEED}:{job['target']}".encode()
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    seed = (SEED + offset) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str})
    metrics = train_task(
        job["target"], records, _FRAME_INDEX, _GPU_ID, job["run_dir"], seed
    )
    return {"target": job["target"], "gpu": _GPU_ID, "metrics": metrics}


def _prepare():
    records_by_target, _, _ = _load_prepared(TARGETS)
    _validate_against_baseline(records_by_target)
    _validate_records(records_by_target)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Classification outputs already exist: {OUTPUT_DIR}")
    if not (WEIGHTS_DIR / PRETRAINED_WEIGHT_FILE).is_file():
        raise FileNotFoundError(WEIGHTS_DIR / PRETRAINED_WEIGHT_FILE)

    prepared, summary, source_hashes = {}, [], {}
    for target, source in records_by_target.items():
        source_path = (
            Path(__file__).resolve().parents[1] / "exp6_face_pair_lab_delta"
            / "outputs" / "task_records" / f"{target}.csv"
        )
        source_hashes[target] = _sha256(source_path)
        if not np.isfinite(source.raw_delta).all():
            raise ValueError(f"Nonfinite laboratory differences: {target}")
        ties = np.isclose(source.raw_delta.to_numpy(float), 0.0, rtol=0, atol=1e-12)
        usable = source.loc[~ties].copy()
        usable["label_up"] = (usable.raw_delta > 0).astype(np.int8)
        if usable.pair_id.duplicated().any():
            raise ValueError(f"Duplicate pair IDs: {target}")
        if usable.groupby("hospital_id").split.nunique().max() != 1:
            raise ValueError(f"Patient leakage: {target}")
        for split in ("train", "val", "test"):
            subset = usable.loc[usable.split.eq(split)]
            if set(subset.label_up.unique()) != {0, 1}:
                raise ValueError(f"Missing direction class: {target}/{split}")
            summary.append({
                "target": target, "split": split,
                "source_pairs": int(source.split.eq(split).sum()),
                "excluded_ties": int((source.split.eq(split) & ties).sum()),
                "pairs": int(len(subset)),
                "patients": int(subset.hospital_id.nunique()),
                "down": int(subset.label_up.eq(0).sum()),
                "up": int(subset.label_up.eq(1).sum()),
            })
        prepared[target] = usable

    OUTPUT_DIR.mkdir(parents=True)
    record_dir = OUTPUT_DIR / "task_records"
    record_dir.mkdir()
    for target, records in prepared.items():
        records.to_csv(record_dir / f"{target}.csv", index=False)
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "label_summary.csv", index=False)
    (OUTPUT_DIR / "experiment_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "experiment": "exp6_face_pair_lab_delta_classifacation",
        "source_experiment": "exp6_face_pair_lab_delta/shared",
        "task": "binary_direction_classification",
        "label": "1 if second laboratory value > first, 0 if second < first",
        "tie_policy": "exclude abs(delta) <= 1e-12",
        "patient_split": "unchanged from source pair records",
        "source_task_record_sha256": source_hashes,
        "source_frame_index_sha256": _sha256(CACHE_DIR / "frame_offsets.npz"),
        "pretrained_weight_sha256": _sha256(WEIGHTS_DIR / PRETRAINED_WEIGHT_FILE),
        "architecture": "shared_siamese_efficientnet_b0_difference_head32",
        "batch_policy": "chunk", "frames_per_video": 20,
        "views": ["original", "hflip", "center_crop", "brightness", "contrast"],
        "loss": "patient-weighted BCEWithLogits; train-pair negative/positive pos_weight",
        "selection": "lowest validation pair-level weighted BCE",
        "decision_threshold": 0.5,
    }, indent=2), encoding="utf-8")
    print(f"[data-ready] targets={len(prepared)} "
          f"source_pairs={sum(row['source_pairs'] for row in summary)} "
          f"excluded_ties={sum(row['excluded_ties'] for row in summary)} "
          f"classification_pairs={sum(row['pairs'] for row in summary)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    _prepare()
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Classification training requires CUDA")
    worker_count = min(args.workers or gpu_count, gpu_count, len(TARGETS))
    jobs = [{
        "target": target,
        "records_path": str(OUTPUT_DIR / "task_records" / f"{target}.csv"),
        "run_dir": str(OUTPUT_DIR / "runs" / target),
    } for target in TARGETS]
    context = mp.get_context("spawn")
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in range(worker_count):
        gpu_queue.put(gpu_id)
    print(f"[scheduler] jobs={len(jobs)} workers={worker_count} "
          f"gpus={gpu_count} dynamic_queue=true", flush=True)
    run_rows, metric_frames, failures = [], [], []
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context,
        initializer=_worker_init,
        initargs=(str(CACHE_DIR / "frame_offsets.npz"), gpu_queue),
    ) as executor:
        futures = {executor.submit(_worker_train, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                metric_frames.append(pd.DataFrame(result["metrics"]))
                run_rows.append({
                    "target": result["target"], "status": "ok",
                    "gpu": result["gpu"], "run_dir": job["run_dir"],
                })
                print(f"[scheduler-complete] target={result['target']} "
                      f"gpu=cuda:{result['gpu']}", flush=True)
            except Exception as exc:
                failure = {
                    "target": job["target"], "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                run_rows.append(failure)
                print(f"[scheduler-failed] target={job['target']} "
                      f"error={failure['error']}", flush=True)
    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "run_index.csv", index=False)
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "metrics_all.csv", index=False
        )
    if failures:
        (OUTPUT_DIR / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} classification jobs failed")
    plot_results(OUTPUT_DIR)
    (OUTPUT_DIR / "COMPLETE").write_text("ok\n", encoding="ascii")
    print("[experiment-complete] all nine tasks and figures generated", flush=True)


if __name__ == "__main__":
    main()
