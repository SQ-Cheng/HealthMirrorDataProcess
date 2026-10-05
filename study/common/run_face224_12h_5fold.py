"""Append three native-224 12-hour five-fold protocols after existing jobs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression import config
from study.exp2_face_pretrained_head32_regression.data import validate_source_data
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.run_patient_diverse_schedule_ablation import STAGE_CONFIG
from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from .rerun_face224 import INDEX_DIR, STATE as FIRST_QUEUE, sha256
from .selected_5fold_splits import FOLDS, TARGETS, prepare_splits
from .selected_5fold_plots import plot_cv, plot_fold_classification, plot_split_distributions
from .run_selected_5fold import _job_seed


STUDY = Path(__file__).resolve().parents[1]
REG = STUDY / "exp2_face_pretrained_head32_regression"
CLASS = STUDY / "exp2_face_pretrained_head32_classification"
SOURCE = REG / "outputs/ablations/lab_match_12h_face224"
STATE = STUDY / "common/outputs/face224_12h_5fold"
SPLITS = STATE / "splits"
PRIOR_ABLATION = FIRST_QUEUE.parent / "patient_diverse_schedule_30_40_face224"
OUTPUTS = {
    "regression_diverse": REG / "outputs/ablations/lab_match_12h_patient_diverse_schedule_30_40_face224/5fold",
    "classification_diverse": CLASS / "outputs/ablations/lab_match_12h_patient_diverse_schedule_30_40_face224/5fold",
    "classification_standard": CLASS / "outputs/ablations/lab_match_12h_face224/5fold",
}
_GPU = _INDEX = None


def schedule_for(protocol):
    schedule = {
        "head_learning_rate": config.HEAD_LEARNING_RATE,
        "head_min_learning_rate": config.MIN_LEARNING_RATE,
        "head_max_epochs": config.HEAD_MAX_EPOCHS,
        "head_patience": config.HEAD_PATIENCE,
        "finetune_learning_rate": config.FINETUNE_LEARNING_RATE,
        "finetune_min_learning_rate": config.MIN_LEARNING_RATE,
        "finetune_max_epochs": config.FINETUNE_MAX_EPOCHS,
        "finetune_patience": config.FINETUNE_PATIENCE,
    }
    if protocol.endswith("diverse"):
        schedule.update(STAGE_CONFIG)
    return schedule


def load_source(target):
    from study.exp2_binary_classification_common.engine import load_task
    path = SOURCE / f"task_records/{target}.csv"
    records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
    original, _ = load_task(target)
    if records.video_id.duplicated().any() or not np.isfinite(records[["raw_value", "abnormal_score", "match_delta_h"]]).all().all():
        raise ValueError(f"Invalid 12h source records: {target}")
    if records.match_delta_h.lt(0).any() or records.match_delta_h.gt(12 + 1e-9).any():
        raise ValueError(f"Records exceed the +/-12h video-interval matching limit: {target}")
    saved = original.set_index("video_id").loc[records.video_id].reset_index()
    identities = ["video_id", "hospital_id", "source_sample_id", "binary_label"]
    pd.testing.assert_frame_equal(records[identities], saved[identities], check_dtype=False, check_exact=True)
    np.testing.assert_allclose(records.raw_value, saved.raw_value, rtol=0, atol=1e-10)
    return records, [sha256(path), sha256(INDEX_DIR / "frame_offsets.npz")]


def preflight():
    if not (SOURCE / "COMPLETE").is_file():
        raise RuntimeError("The native-224 12h regression source is incomplete")
    validate_source_data(SOURCE / "source_data", expected_max_delta_hours=12)
    manifest = json.loads((SOURCE / "experiment_manifest.json").read_text())
    if manifest["frame_index_sha256"] != sha256(INDEX_DIR / "frame_offsets.npz"):
        raise RuntimeError("12h source frame index has changed")
    index = FrameOffsetIndex.load(INDEX_DIR / "frame_offsets.npz")
    if set(index.video_formats) != {"ffv1"} or not _index_is_reusable(INDEX_DIR, index.video_ids, "20frame"):
        raise RuntimeError("The shared native FFV1 index is stale")
    counts, fingerprints = [], {}
    for target in TARGETS:
        records, hashes = load_source(target)
        if not set(records.video_id).issubset(index.video_lookup) or any(
            index.frame_range(video)[1] - index.frame_range(video)[0] != 20 for video in records.video_id
        ):
            raise RuntimeError(f"20-frame native source coverage is missing: {target}")
        fingerprints[target] = hashes
        counts.append({"target": target, "videos": len(records), "patients": records.hospital_id.nunique()})
    from study.exp2_face_pretrained_head32_regression.models import WEIGHT_FILES
    fingerprints["pretrained_weight"] = sha256(Path(config.WEIGHTS_DIR) / WEIGHT_FILES["efficientnet_b0"])
    print(f"[preflight-ok] 224/12h/20frames/5views targets={len(TARGETS)}", flush=True)
    return fingerprints, pd.DataFrame(counts)


def dependency_finished(directory, lock_name):
    with (directory / lock_name).open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (directory / "COMPLETE").is_file():
            raise RuntimeError(f"Preceding job stopped before completion: {directory}")
    return True


def validate_splits():
    scalers = json.loads((SPLITS / "scalers.json").read_text())
    assignments = pd.read_csv(SPLITS / "patient_folds.csv", dtype={"hospital_id": str})
    for target in TARGETS:
        source, _ = load_source(target)
        lookup = assignments.loc[assignments.target.eq(target)].set_index("hospital_id", verify_integrity=True).fold
        coverage, patient_roles = [], {}
        for fold in range(FOLDS):
            records = pd.read_csv(SPLITS / f"{target}_fold{fold}.csv", dtype={"hospital_id": str, "video_id": str})
            ordered = records.set_index("video_id").loc[source.video_id].reset_index()
            identity = ["video_id", "hospital_id", "binary_label", "source_sample_id"]
            pd.testing.assert_frame_equal(ordered[identity], source[identity], check_dtype=False)
            np.testing.assert_allclose(ordered.raw_value, source.raw_value, rtol=0, atol=1e-10)
            if len(records) != len(source) or set(records.split) != {"train", "val", "test"} or records.groupby("hospital_id").split.nunique().max() != 1:
                raise AssertionError(f"Patient leakage or cohort change: {target}/{fold}")
            assigned = records.hospital_id.map(lookup)
            expected_roles = np.where(assigned.eq(fold), "test", np.where(assigned.eq((fold + 1) % FOLDS), "val", "train"))
            if assigned.isna().any() or not np.array_equal(records.split, expected_roles):
                raise AssertionError(f"Patient folds or cyclic validation roles changed: {target}/{fold}")
            for split, group in records.groupby("split"):
                if group.binary_label.nunique() != 2:
                    raise AssertionError(f"Single-class fold: {target}/{fold}/{split}")
            fitted = fit_robust_target_scaler(target, records, config.SCORE_DEFINITIONS[target]["unit"]).to_dict()
            if fitted != scalers[target][str(fold)]:
                raise AssertionError(f"Scaler was not fitted on this fold's training rows: {target}/{fold}")
            np.testing.assert_allclose(RobustTargetScaler(**fitted).transform(records.raw_value), records.robust_scaled_raw_value, rtol=0, atol=1e-10)
            coverage.extend(records.loc[records.split.eq("test"), "video_id"])
            for patient in records.loc[records.split.eq("test"), "hospital_id"].unique():
                patient_roles[patient] = patient_roles.get(patient, 0) + 1
        if set(coverage) != set(source.video_id) or len(coverage) != len(source) or set(patient_roles.values()) != {1}:
            raise AssertionError(f"OOF coverage is not exactly once per video/patient: {target}")
    return scalers


def worker_init(queue):
    global _GPU, _INDEX
    _GPU = int(queue.get())
    torch.cuda.set_device(_GPU)
    torch.set_num_threads(2)
    _INDEX = FrameOffsetIndex.load(INDEX_DIR / "frame_offsets.npz")


def run_worker(job):
    protocol, target, fold = job["protocol"], job["target"], job["fold"]
    seed = _job_seed(target, fold)
    path = SPLITS / f"{target}_fold{fold}.csv"
    directory = OUTPUTS[protocol] / f"fold_{fold}"
    run = directory / f"runs/efficientnet_b0/{target}"
    schedule = schedule_for(protocol)
    policy = "patient_diverse" if protocol.endswith("diverse") else "chunked"
    if protocol == "regression_diverse":
        from study.exp2_face_pretrained_head32_regression.train import train_task
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_task(
            "efficientnet_b0", target, _INDEX,
            pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}),
            RobustTargetScaler(**job["scaler"]), config.WEIGHTS_DIR, str(run),
            train_batch_policy=policy,
            head_epochs=schedule["head_max_epochs"], head_patience=schedule["head_patience"],
            head_learning_rate=schedule["head_learning_rate"], head_min_learning_rate=schedule["head_min_learning_rate"],
            finetune_epochs=schedule["finetune_max_epochs"], finetune_patience=schedule["finetune_patience"],
            finetune_learning_rate=schedule["finetune_learning_rate"], finetune_min_learning_rate=schedule["finetune_min_learning_rate"],
        )
    else:
        from study.exp2_binary_classification_common.engine import train_task
        train_task("face_only", target, _GPU, seed, output_dir=directory,
                   train_batch_policy=policy, stage_config=schedule,
                   records_path=path, frame_index_path=INDEX_DIR / "frame_offsets.npz")
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["target"] != target or checkpoint["train_batch_policy"] != policy:
        raise RuntimeError(f"Wrong saved checkpoint: {run}")
    if protocol == "regression_diverse":
        if (any(checkpoint[key] != value for key, value in schedule.items())
                or checkpoint["target_scaler"] != job["scaler"]
                or not checkpoint.get("model_state_dict")):
            raise RuntimeError(f"Regression checkpoint schedule/scaler is incorrect: {run}")
    elif checkpoint["stage_config"] != schedule or checkpoint["seed"] != seed or not checkpoint.get("state_dict"):
        raise RuntimeError(f"Binary checkpoint schedule/seed is incorrect: {run}")
    (run / "job_complete.json").write_text(json.dumps({"contract": job["contract"], "seed": seed}))
    return {"fold": fold, "target": target, "seed": seed, "gpu": _GPU, "architecture": "efficientnet_b0", "status": "ok"}


def run_protocol(protocol, scalers):
    root = OUTPUTS[protocol]
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": protocol, "folds": FOLDS, "targets": list(TARGETS),
        "source": str(SOURCE), "source_resolution": 224, "matching_hours": 12,
        "split_manifest_sha256": sha256(SPLITS / "manifest.json"),
        "split_records_sha256": {f"{t}/{f}": sha256(SPLITS / f"{t}_fold{f}.csv") for t in TARGETS for f in range(FOLDS)},
        "frame_index_sha256": sha256(INDEX_DIR / "frame_offsets.npz"),
        "pretrained_weight_sha256": preflight()[0]["pretrained_weight"],
        "stage_config": schedule_for(protocol), "views": list(config.VIEW_NAMES),
        "frames_per_video": 20, "head_hidden_features": 32, "architecture": "efficientnet_b0",
        "batch_policy": "patient_diverse" if protocol.endswith("diverse") else "chunked",
        "train_source_batch_size": config.TRAIN_SOURCE_BATCH_SIZES["efficientnet_b0"],
        "optimizer": "AdamW", "weight_decay": config.WEIGHT_DECAY,
        "compile": config.TORCH_COMPILE_ENABLED, "compile_mode": config.TORCH_COMPILE_MODE,
        "seed_policy": "same target/fold hash-derived seed for all three protocols",
        "split_policy": "test=k, val=(k+1)%5, train=other three; shared patient-disjoint folds",
        "loss": "SmoothL1(beta=0.5), fold-train robust scaling" if protocol == "regression_diverse" else "BCEWithLogitsLoss; fold-train negative/positive video count weight",
    }
    path = root / "experiment_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError(f"Existing protocol differs: {root}")
    path.write_text(json.dumps(manifest, indent=2))
    contract = sha256(path)
    if (root / "COMPLETE").is_file():
        print(f"[reuse-complete] {protocol}", flush=True)
        return
    jobs = [{"protocol": protocol, "target": target, "fold": fold, "scaler": scalers[target][str(fold)], "contract": contract}
            for fold in range(FOLDS) for target in TARGETS]
    rows, pending = [], []
    for job in jobs:
        run = root / f"fold_{job['fold']}/runs/efficientnet_b0/{job['target']}"
        marker = run / "job_complete.json"
        if marker.is_file() and json.loads(marker.read_text()) == {"contract": contract, "seed": _job_seed(job['target'], job['fold'])} and all((run / f).is_file() for f in ("model.pt", "history.csv", "metrics.csv", "video_predictions.csv")):
            rows.append({"fold": job["fold"], "target": job["target"], "architecture": "efficientnet_b0", "status": "ok", "gpu": "reused"})
        else:
            pending.append(job)
    workers = min(4, torch.cuda.device_count(), len(pending))
    if pending and workers < 1:
        raise RuntimeError("Five-fold training requires CUDA")
    print(f"[protocol-start] {protocol} jobs=40 pending={len(pending)} gpus={workers}", flush=True)
    ctx = mp.get_context("spawn")
    failures = []
    if pending:
        with ctx.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers):
                queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=worker_init, initargs=(queue,)) as pool:
                futures = {pool.submit(run_worker, job): job for job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        row = future.result()
                    except Exception:
                        row = {"fold": job["fold"], "target": job["target"], "status": "failed", "architecture": "efficientnet_b0", "error": traceback.format_exc()}
                        failures.append(row)
                    rows.append(row)
                    directory = root / f"fold_{job['fold']}"
                    directory.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame([r for r in rows if r["fold"] == job["fold"]]).to_csv(directory / "run_index.csv", index=False)
                    print(f"[task-finished] {protocol}/{job['target']}/fold{job['fold']} {row['status']}", flush=True)
    if failures:
        (root / "failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError(f"{len(failures)} five-fold jobs failed: {protocol}")
    for fold in range(FOLDS):
        directory = root / f"fold_{fold}"
        pd.DataFrame([r for r in rows if r["fold"] == fold]).to_csv(directory / "run_index.csv", index=False)
        runs = [directory / f"runs/efficientnet_b0/{target}" for target in TARGETS]
        for name in ("metrics", "history"):
            pd.concat([pd.read_csv(r / f"{name}.csv") for r in runs], ignore_index=True).to_csv(directory / f"{name}_all.csv", index=False)
        if protocol == "regression_diverse":
            from study.exp2_face_pretrained_head32_regression.plot_results import main as plot
            plot(directory)
        else:
            plot_fold_classification(directory)
        (directory / "COMPLETE").write_text("ok\n")
    plot_cv(root, "face_regression" if protocol == "regression_diverse" else "face_classification", split_root=SPLITS)
    from .plot_face224_12h_5fold import plot_oof
    plot_oof(root, protocol)
    (root / "COMPLETE").write_text("all five folds and OOF figures completed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--protocol", choices=tuple(OUTPUTS), help=argparse.SUPPRESS)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    os.environ["HEALTHMIRROR_FACE_SOURCE"] = "face224"
    fingerprints, counts = preflight()
    if args.check_only:
        print(counts.to_string(index=False))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    if args.prepare_only or args.protocol:
        prepare_splits(load_source, SPLITS, source_policy={"source": str(SOURCE), "matching_hours": 12, "resolution": 224})
        scalers = validate_splits()
        if args.prepare_only:
            plot_split_distributions(SPLITS)
            print("[prepare-complete] shared balanced folds checked; no training started", flush=True)
        else:
            run_protocol(args.protocol, scalers)
        return
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {"outputs": {k: str(v) for k, v in OUTPUTS.items()}, "source_sha256": fingerprints}
        plan_path = STATE / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("The registered 12h five-fold source or plan changed")
        plan_path.write_text(json.dumps(plan, indent=2))
        counts.to_csv(STATE / "source_counts.csv", index=False)
        while not (dependency_finished(FIRST_QUEUE, ".queue.lock") and dependency_finished(PRIOR_ABLATION, ".monitor.lock")):
            print("[waiting] existing 224 queue / patient-diverse ablation active; no GPU allocated", flush=True)
            time.sleep(args.poll_seconds)
        if preflight()[0] != fingerprints:
            raise RuntimeError("12h source changed while waiting")
        prepare_splits(load_source, SPLITS, source_policy={"source": str(SOURCE), "matching_hours": 12, "resolution": 224})
        validate_splits()
        plot_split_distributions(SPLITS)
        for protocol, output in OUTPUTS.items():
            logs = Path(str(output).replace("/outputs/", "/logs/", 1))
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "run.log").open("a") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "study.common.run_face224_12h_5fold", "--protocol", protocol], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if child.wait():
                    raise RuntimeError(f"Five-fold queue stopped: {protocol}")
        from .plot_face224_12h_5fold import plot_classification_comparison
        plot_classification_comparison(OUTPUTS, STATE / "figures")
        if preflight()[0] != fingerprints:
            raise RuntimeError("12h reference changed during training")
        (STATE / "COMPLETE").write_text("three five-fold experiments and comparisons completed\n")
        print("[queue-complete] native224 12h five-fold experiments", flush=True)


if __name__ == "__main__":
    main()
