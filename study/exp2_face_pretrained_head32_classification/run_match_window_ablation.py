"""Train binary 12/6-hour matching-window variants on the saved patient splits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_lab_multimodal.config import SEED
from study.exp2_binary_classification_common.engine import (
    FRAME_INDEX_PATH, PREPARED_DIR, REFERENCE_DIR, TARGETS, train_task,
)
from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.models import WEIGHT_FILES

from .plot_match_window_comparison import plot_comparison


HERE = Path(__file__).resolve().parent
REGRESSION_ABLATIONS = HERE.parent / "exp2_face_pretrained_head32_regression/outputs/ablations"
BASELINE = HERE / "outputs"
_GPU = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(hours):
    source = REGRESSION_ABLATIONS / f"lab_match_{hours}h"
    output = BASELINE / "ablations" / f"lab_match_{hours}h"
    if output.exists():
        raise FileExistsError(output)
    if not (source / "COMPLETE").is_file():
        raise RuntimeError(f"Matching-window source is incomplete: {source}")
    source_manifest = json.loads((source / "experiment_manifest.json").read_text())
    if (source_manifest["lab_match_max_delta_hours"] != hours
            or source_manifest["result_variant"] != "20frame"):
        raise AssertionError("Unexpected matching-window source policy")
    baseline_index = pd.read_csv(BASELINE / "run_index.csv")
    if (len(baseline_index) != len(TARGETS)
            or set(baseline_index.target) != set(TARGETS)
            or not baseline_index.status.eq("complete").all()):
        raise RuntimeError("24-hour classification baseline is incomplete")
    frame_index = FrameOffsetIndex.load(FRAME_INDEX_PATH)
    weight_path = HERE.parent / "common/pretrained_weights" / WEIGHT_FILES["efficientnet_b0"]
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    sources = {}
    for target in TARGETS:
        path = source / "task_records" / f"{target}.csv"
        candidate = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
        baseline_path = (PREPARED_DIR if target == "total_bilirubin_high" else REFERENCE_DIR) / "task_records" / f"{target}.csv"
        original = pd.read_csv(baseline_path, dtype={"hospital_id": str, "video_id": str})
        filtered = original.loc[original.match_delta_h.le(hours)].sort_values("video_id").reset_index(drop=True)
        candidate = candidate.sort_values("video_id").reset_index(drop=True)
        columns = ["hospital_id", "video_id", "source_sample_id", "binary_label",
                   "raw_value"]
        pd.testing.assert_frame_equal(candidate[columns], filtered[columns],
                                      check_dtype=False, check_exact=True)
        if not np.allclose(candidate.match_delta_h, filtered.match_delta_h,
                           rtol=0, atol=1e-9):
            raise AssertionError(f"Source match time differs: {target}")
        if (candidate.video_id.duplicated().any()
                or set(candidate.split) != {"train", "val", "test"}
                or candidate.groupby("hospital_id").split.nunique().max() != 1):
            raise AssertionError(f"Invalid candidate split: {target}")
        for split, group in candidate.groupby("split"):
            if set(group.binary_label.astype(int)) != {0, 1}:
                raise AssertionError(f"Single-class {split} split: {target}")
        if any(frame_index.frame_range(video_id)[1] - frame_index.frame_range(video_id)[0] != 20
               for video_id in candidate.video_id):
            raise AssertionError(f"Missing 20-frame index coverage: {target}")
        sources[target] = {"path": str(path.resolve()), "sha256": _sha256(path),
                           "videos": len(candidate)}
        print(f"[source-validated] hours={hours} task={target} videos={len(candidate)}", flush=True)
    return source, output, sources, weight_path


def _worker_init(queue):
    global _GPU
    _GPU = int(queue.get())
    torch.cuda.set_device(_GPU)
    print(f"[worker-ready] gpu=cuda:{_GPU}", flush=True)


def _worker(job):
    metrics = train_task(
        "face_only", job["target"], _GPU, SEED,
        output_dir=job["output"], records_path=job["records_path"],
    )
    return metrics, _GPU


def run(hours):
    source, output, sources, weight_path = preflight(hours)
    gpu_count = min(4, torch.cuda.device_count(), len(TARGETS))
    if gpu_count < 1:
        raise RuntimeError("CUDA is required for classification training")
    output.mkdir(parents=True)
    (output / "experiment_manifest.json").write_text(json.dumps({
        "schema_version": 1, "task_type": "true_binary_classification",
        "matching_window_hours": hours, "source_output": str(source),
        "baseline_output": str(BASELINE), "seed": SEED,
        "architecture": "efficientnet_b0_head32", "targets": list(TARGETS),
        "task_records": sources, "frame_index_sha256": _sha256(FRAME_INDEX_PATH),
        "pretrained_weight_sha256": _sha256(weight_path),
        "split_policy": "reuse independently searched patient-disjoint window split",
        "training_policy": "same as 24h classification baseline; train-only video-count pos_weight",
        "comparison_policy": "report full test cohorts and exact shared held-out videos separately",
    }, indent=2), encoding="utf-8")
    context = mp.get_context("spawn")
    manager = context.Manager()
    queue = manager.Queue()
    for gpu in range(gpu_count):
        queue.put(gpu)
    jobs = [{"target": target, "records_path": sources[target]["path"],
             "output": str(output)} for target in TARGETS]
    rows, metrics, failures = [], [], []
    print(f"[scheduler-start] hours={hours} jobs={len(jobs)} gpus={gpu_count}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=gpu_count, mp_context=context,
                                 initializer=_worker_init, initargs=(queue,)) as executor:
            futures = {executor.submit(_worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result, gpu = future.result()
                    metrics.append(pd.DataFrame(result))
                    row = {"target": job["target"], "status": "complete", "gpu": gpu}
                except Exception as exc:
                    row = {"target": job["target"], "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                    failures.append(row)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
                if metrics:
                    pd.concat(metrics, ignore_index=True).to_csv(
                        output / "metrics_all.csv", index=False)
                print(f"[scheduler] hours={hours} task={job['target']} status={row['status']}", flush=True)
    finally:
        manager.shutdown()
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} classification jobs failed for {hours}h")
    plot_comparison(BASELINE, output, TARGETS, hours)
    (output / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[window-ablation-complete] hours={hours} output={output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, choices=(6, 12), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        preflight(args.hours)
        print(f"[preflight-complete] hours={args.hours}", flush=True)
    else:
        run(args.hours)


if __name__ == "__main__":
    main()
