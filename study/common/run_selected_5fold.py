"""Five-fold training for exactly four selected face/video experiments."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_binary_classification_common.engine import train_task as train_binary
from study.exp2_face_pretrained_head32_regression import config as face_config
from study.exp2_face_pretrained_head32_regression.plot_results import main as plot_face
from study.exp2_face_pretrained_head32_regression.run_patient_diverse_schedule_ablation import (
    STAGE_CONFIG,
)
from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler
from study.exp2_face_pretrained_head32_regression.train import train_task as train_face
from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex
from study.exp3_video_lab_regression.clips import ClipIndex
from study.exp3_video_lab_regression.config import INDEX_DIR as VIDEO_INDEX_DIR
from study.exp3_video_lab_regression.plot_results import plot_results as plot_video
from study.exp3_video_lab_regression.prepare import prepare as prepare_video
from study.exp3_video_lab_regression.train import train_task as train_video

from .selected_5fold_plots import (plot_cv, plot_fold_classification,
                                   plot_regression_comparison, plot_split_distributions)
from .selected_5fold_splits import (FOLDS, ROOT, SEED, SPLIT_ROOT, TARGETS,
                                    prepare_splits)


PROTOCOLS = (
    "face_regression", "face_regression_diverse_30_40",
    "video_regression", "face_classification",
)
OUTPUTS = {
    "face_regression": ROOT / "study/exp2_face_pretrained_head32_regression/outputs/5fold",
    "face_regression_diverse_30_40": ROOT / "study/exp2_face_pretrained_head32_regression/outputs/ablations/patient_diverse_schedule_30_40/5fold",
    "video_regression": ROOT / "study/exp3_video_lab_regression/outputs/5fold",
    "face_classification": ROOT / "study/exp2_face_pretrained_head32_classification/outputs/5fold",
}
FACE_INDEX_PATH = Path(face_config.REFERENCE_INDEX_DIR) / "frame_offsets.npz"
VIDEO_INDEX_PATH = VIDEO_INDEX_DIR / "clip_offsets.npz"
_GPU_ID = None
_INDEX = None
_PROTOCOL = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _job_seed(target, fold):
    token = f"{SEED}:efficientnet_b0:{target}:fold{fold}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little") % (2**31 - 1)


def _run_dir(directory, target, protocol):
    root = directory / "runs"
    return root / target if protocol == "video_regression" else root / "efficientnet_b0" / target


def _worker_init(protocol, index_path, gpu_queue):
    global _GPU_ID, _INDEX, _PROTOCOL
    _GPU_ID = int(gpu_queue.get())
    _PROTOCOL = protocol
    torch.cuda.set_device(_GPU_ID)
    torch.set_num_threads(2)
    if protocol in ("face_regression", "face_regression_diverse_30_40"):
        _INDEX = FrameOffsetIndex.load(index_path)
    elif protocol == "video_regression":
        _INDEX = ClipIndex.load(index_path)
    print(f"[worker-ready] protocol={protocol} gpu=cuda:{_GPU_ID}", flush=True)


def _worker(job):
    target, fold, protocol = job["target"], job["fold"], _PROTOCOL
    records_path = SPLIT_ROOT / f"{target}_fold{fold}.csv"
    fold_dir = OUTPUTS[protocol] / f"fold_{fold}"
    seed = _job_seed(target, fold)
    if protocol in ("face_regression", "face_regression_diverse_30_40"):
        records = pd.read_csv(records_path,
                              dtype={"hospital_id": str, "video_id": str})
        scaler = RobustTargetScaler(**job["scaler"])
        options = {}
        if protocol == "face_regression_diverse_30_40":
            options = {
                "train_batch_policy": "patient_diverse",
                "head_learning_rate": STAGE_CONFIG["head_learning_rate"],
                "head_min_learning_rate": face_config.MIN_LEARNING_RATE,
                "head_epochs": STAGE_CONFIG["head_max_epochs"],
                "head_patience": STAGE_CONFIG["head_patience"],
                "finetune_learning_rate": STAGE_CONFIG["finetune_learning_rate"],
                "finetune_min_learning_rate": STAGE_CONFIG["finetune_min_learning_rate"],
                "finetune_epochs": STAGE_CONFIG["finetune_max_epochs"],
                "finetune_patience": STAGE_CONFIG["finetune_patience"],
            }
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_face(
            architecture="efficientnet_b0", target=target,
            frame_index=_INDEX, records=records, target_scaler=scaler,
            weights_dir=face_config.WEIGHTS_DIR,
            run_dir=str(_run_dir(fold_dir, target, protocol)),
            **options,
        )
    elif protocol == "video_regression":
        records = pd.read_csv(records_path,
                              dtype={"hospital_id": str, "video_id": str})
        train_video(
            target, records, job["scaler"], _INDEX, _GPU_ID,
            _run_dir(fold_dir, target, protocol), seed,
            variant="selected_5fold",
        )
    else:
        train_binary(
            "face_only", target, _GPU_ID, seed,
            output_dir=fold_dir, records_path=records_path,
        )
    return {"target": target, "fold": fold, "seed": seed, "gpu": _GPU_ID}


def _finalize_fold(protocol, fold):
    directory = OUTPUTS[protocol] / f"fold_{fold}"
    rows = pd.read_csv(directory / "run_index.csv")
    if (len(rows) != len(TARGETS) or set(rows.target) != set(TARGETS)
            or not rows.status.eq("ok").all()):
        raise RuntimeError(f"Incomplete fold: {directory}")
    metrics, histories = [], []
    for target in TARGETS:
        run_dir = _run_dir(directory, target, protocol)
        metrics.append(pd.read_csv(run_dir / "metrics.csv"))
        histories.append(pd.read_csv(run_dir / "history.csv"))
    pd.concat(metrics, ignore_index=True).to_csv(directory / "metrics_all.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(directory / "history_all.csv", index=False)
    if protocol in ("face_regression", "face_regression_diverse_30_40"):
        plot_face(directory)
    elif protocol == "video_regression":
        plot_video(directory, reference_dir=OUTPUTS["face_regression"] / f"fold_{fold}")
    else:
        plot_fold_classification(directory)
    (directory / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[fold-complete] protocol={protocol} fold={fold}", flush=True)


def _run_protocol(protocol, scalers):
    root = OUTPUTS[protocol]
    if (root / "COMPLETE").is_file():
        print(f"[protocol-reused] protocol={protocol}", flush=True)
        return
    if protocol == "video_regression" and not OUTPUTS["face_regression"].joinpath("COMPLETE").is_file():
        raise RuntimeError("Video fold plotting requires completed face-regression folds")
    root.mkdir(parents=True, exist_ok=True)
    index_path = (
        VIDEO_INDEX_PATH if protocol == "video_regression" else FACE_INDEX_PATH
        if protocol != "face_classification" else None
    )
    manifest = {
        "schema_version": 1, "protocol": protocol, "folds": FOLDS,
        "targets": list(TARGETS), "architecture": (
            "kinetics_r3d18_head32" if protocol == "video_regression"
            else "imagenet_efficientnet_b0_head32"
        ),
        "split_manifest_sha256": _sha256(SPLIT_ROOT / "manifest.json"),
        "split_policy": "patient-group five-fold; test=k, val=(k+1)%5, train=other 3",
        "job_seed_policy": "sha256(base_seed:efficientnet_b0:target:fold)",
        "training_change": (
            "patient-diverse batches and Exp2 patient_diverse_schedule_30_40"
            if protocol == "face_regression_diverse_30_40"
            else "only shared 5-fold split, per-fold train-only scaler/pos_weight"
        ),
        "index_path": str(index_path) if index_path else None,
    }
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(f"Existing protocol contract differs: {root}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs = [{"target": target, "fold": fold,
             "scaler": scalers[target][str(fold)]}
            for fold in range(FOLDS) for target in TARGETS]
    ready = []
    for job in jobs:
        directory = OUTPUTS[protocol] / f"fold_{job['fold']}"
        run_dir = _run_dir(directory, job["target"], protocol)
        required = ["metrics.csv", "history.csv", "model.pt", "video_predictions.csv"]
        if protocol in ("video_regression", "face_classification"):
            required.append("run_manifest.json")
        if all((run_dir / name).is_file()
               for name in required):
            ready.append(job)
    pending = [job for job in jobs if job not in ready]
    print(f"[protocol-start] protocol={protocol} total={len(jobs)} "
          f"reused={len(ready)} pending={len(pending)}", flush=True)
    results = {(job["fold"], job["target"]): {
        "target": job["target"], "fold": job["fold"], "status": "ok",
        "seed": _job_seed(job["target"], job["fold"]), "gpu": "reused",
        "architecture": "efficientnet_b0",
    } for job in ready}
    failures = []
    if pending:
        count = min(4, torch.cuda.device_count(), len(pending))
        if count < 1:
            raise RuntimeError("5-fold training requires CUDA")
        context = mp.get_context("spawn")
        manager = context.Manager()
        queue = manager.Queue()
        for gpu_id in range(count):
            queue.put(gpu_id)
        with ProcessPoolExecutor(
            max_workers=count, mp_context=context,
            initializer=_worker_init,
            initargs=(protocol, str(index_path) if index_path else None, queue),
        ) as executor:
            futures = {executor.submit(_worker, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    result.update(status="ok", architecture="efficientnet_b0")
                    print(f"[job-complete] protocol={protocol} "
                          f"fold={job['fold']} target={job['target']}", flush=True)
                except Exception as exc:
                    result = {
                        "target": job["target"], "fold": job["fold"],
                        "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "architecture": "efficientnet_b0",
                    }
                    failures.append(result)
                    print(f"[job-failed] protocol={protocol} fold={job['fold']} "
                          f"target={job['target']} error={exc}", flush=True)
                results[(job["fold"], job["target"])] = result
                directory = root / f"fold_{job['fold']}"
                directory.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row for (fold, _), row in results.items()
                              if fold == job["fold"]]).to_csv(
                    directory / "run_index.csv", index=False
                )
    for fold in range(FOLDS):
        directory = root / f"fold_{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row for (current, _), row in results.items()
                      if current == fold]).to_csv(directory / "run_index.csv", index=False)
    if failures:
        (root / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} jobs failed in {protocol}")
    for fold in range(FOLDS):
        if not (root / f"fold_{fold}" / "COMPLETE").is_file():
            _finalize_fold(protocol, fold)
    plot_cv(root, protocol)
    (root / "COMPLETE").write_text("ok\n", encoding="ascii")
    print(f"[protocol-complete] protocol={protocol}", flush=True)


def main():
    if not (ROOT / "study/exp3_video_lab_regression/outputs/COMPLETE").is_file():
        raise RuntimeError("Exp3 baseline is incomplete")
    prepare_video()
    prepare_splits()
    plot_split_distributions()
    with open(SPLIT_ROOT / "scalers.json", encoding="utf-8") as handle:
        scalers = json.load(handle)
    for protocol in PROTOCOLS:
        _run_protocol(protocol, scalers)
    plot_regression_comparison(
        tuple(OUTPUTS[name] for name in PROTOCOLS[:3]),
        OUTPUTS["face_regression"] / "figures/comparison",
    )
    print("[selected-5fold-complete] four specified protocols", flush=True)


if __name__ == "__main__":
    main()
