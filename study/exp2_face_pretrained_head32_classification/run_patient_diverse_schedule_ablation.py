"""Run the face-only binary patient-diverse 30/40 ablation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import pandas as pd
import torch

from study.exp2_lab_multimodal.config import SEED
from study.exp2_binary_classification_common.engine import (
    EXPERIMENT_DIRS, FRAME_INDEX_PATH, PREPARED_DIR, REFERENCE_DIR,
    TARGETS, WEIGHTS_DIR, train_task,
)
from study.exp2_face_history_head32_regression import config as base_config
from study.exp2_face_pretrained_head32_regression import config as face_config
from study.exp2_face_pretrained_head32_regression.models import WEIGHT_FILES
from study.exp2_face_pretrained_head32_regression.run_patient_diverse_schedule_ablation import (
    STAGE_CONFIG,
)

from .plot_patient_diverse_schedule_comparison import plot_comparison


BASE_DIR = EXPERIMENT_DIRS["face_only"] / "outputs"
VARIANT_DIR = BASE_DIR / "ablations" / "patient_diverse_schedule_30_40"
_GPU_ID = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight():
    if VARIANT_DIR.exists():
        raise FileExistsError(f"Ablation output already exists: {VARIANT_DIR}")
    index = pd.read_csv(BASE_DIR / "run_index.csv")
    if (len(index) != len(TARGETS) or set(index.target) != set(TARGETS)
            or not index.status.eq("complete").all()):
        raise RuntimeError("Face-only binary baseline is incomplete")
    if not FRAME_INDEX_PATH.is_file():
        raise FileNotFoundError(FRAME_INDEX_PATH)
    weight_path = Path(WEIGHTS_DIR) / WEIGHT_FILES["efficientnet_b0"]
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    sources = {}
    for target in TARGETS:
        source = PREPARED_DIR if target == "total_bilirubin_high" else REFERENCE_DIR
        path = source / "task_records" / f"{target}.csv"
        records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
        baseline_dir = BASE_DIR / "runs" / "efficientnet_b0" / target
        reference = pd.read_csv(
            baseline_dir / "video_predictions.csv",
            dtype={"hospital_id": str, "video_id": str},
        )
        manifest = json.loads((baseline_dir / "run_manifest.json").read_text())
        if manifest["seed"] != SEED:
            raise AssertionError(f"Baseline seed mismatch: {target}")
        if len(records) != len(reference) or records.video_id.duplicated().any():
            raise AssertionError(f"Baseline video count or uniqueness mismatch: {target}")
        records = records.sort_values("video_id").reset_index(drop=True)
        reference = reference.sort_values("video_id").reset_index(drop=True)
        for column in ("hospital_id", "video_id", "split"):
            if not records[column].astype(str).equals(reference[column].astype(str)):
                raise AssertionError(f"Baseline {column} mismatch: {target}")
        if not records.binary_label.astype(int).equals(reference.y_true.astype(int)):
            raise AssertionError(f"Baseline binary labels mismatch: {target}")
        sources[target] = {"path": str(path.resolve()), "sha256": _sha256(path),
                           "videos": int(len(records))}
    schedule = {
        "head_learning_rate": base_config.HEAD_LEARNING_RATE,
        "head_min_learning_rate": base_config.MIN_LEARNING_RATE,
        "head_max_epochs": base_config.HEAD_MAX_EPOCHS,
        "head_patience": base_config.HEAD_PATIENCE,
        "finetune_learning_rate": base_config.FINETUNE_LEARNING_RATE,
        "finetune_min_learning_rate": base_config.MIN_LEARNING_RATE,
        "finetune_max_epochs": base_config.FINETUNE_MAX_EPOCHS,
        "finetune_patience": base_config.FINETUNE_PATIENCE,
    }
    schedule.update(STAGE_CONFIG)
    expected = {
        "head_learning_rate": 1e-4, "head_min_learning_rate": 1e-6,
        "head_max_epochs": 30, "head_patience": 8,
        "finetune_learning_rate": 3e-6, "finetune_min_learning_rate": 1e-7,
        "finetune_max_epochs": 40, "finetune_patience": 8,
    }
    if schedule != expected:
        raise AssertionError(f"The resolved 30/40 schedule differs: {schedule}")
    return sources, schedule, weight_path


def _worker_init(gpu_queue):
    global _GPU_ID
    _GPU_ID = int(gpu_queue.get())
    torch.cuda.set_device(_GPU_ID)
    print(f"[worker-ready] gpu=cuda:{_GPU_ID}", flush=True)


def _worker(job):
    metrics = train_task(
        "face_only", job["target"], _GPU_ID, SEED,
        output_dir=VARIANT_DIR, train_batch_policy="patient_diverse",
        stage_config=STAGE_CONFIG,
    )
    return {"target": job["target"], "gpu": _GPU_ID, "metrics": metrics}


def main():
    sources, schedule, weight_path = _preflight()
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Ablation requires CUDA")
    VARIANT_DIR.mkdir(parents=True)
    (VARIANT_DIR / "experiment_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "experiment": "exp2_face_only_binary_patient_diverse_schedule_30_40",
        "baseline_output": str(BASE_DIR.resolve()),
        "modality": "face_only", "architecture": "efficientnet_b0",
        "seed": SEED, "targets": list(TARGETS),
        "task_records": sources,
        "frame_index_sha256": _sha256(FRAME_INDEX_PATH),
        "pretrained_weight_sha256": _sha256(weight_path),
        "training_views": list(base_config.VIEW_NAMES),
        "frames_per_video": base_config.FRAMES_PER_VIDEO,
        "train_batch_policy": "patient_diverse",
        "source_batch_size": face_config.TRAIN_SOURCE_BATCH_SIZES["efficientnet_b0"],
        "weight_decay": base_config.WEIGHT_DECAY,
        "training_protocol": "two_stage_full",
        "stage_config": schedule,
        "unchanged": ["binary labels", "patient split", "frame index",
                      "five views", "model", "pos_weight", "seed",
                      "validation selection", "test evaluation"],
    }, indent=2), encoding="utf-8")
    context = mp.get_context("spawn")
    worker_count = min(4, gpu_count, len(TARGETS))
    manager = context.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in range(worker_count):
        gpu_queue.put(gpu_id)
    jobs = [{"target": target} for target in TARGETS]
    print(f"[scheduler] jobs={len(jobs)} workers={worker_count} "
          f"gpus={gpu_count} patient_diverse=true", flush=True)
    rows, metrics, failures = [], [], []
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context,
        initializer=_worker_init, initargs=(gpu_queue,),
    ) as executor:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                metrics.append(pd.DataFrame(result["metrics"]))
                rows.append({"target": job["target"], "status": "complete",
                             "gpu": result["gpu"]})
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
    pd.DataFrame(rows).to_csv(VARIANT_DIR / "run_index.csv", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(
            VARIANT_DIR / "metrics_all.csv", index=False
        )
    if failures:
        (VARIANT_DIR / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} classification jobs failed")
    plot_comparison(BASE_DIR, VARIANT_DIR, TARGETS)
    (VARIANT_DIR / "COMPLETE").write_text("ok\n", encoding="ascii")
    print("[experiment-complete] all eight jobs and comparison figures generated",
          flush=True)


if __name__ == "__main__":
    main()
