"""Resume prepared native-224 protocols without any legacy baseline dependency."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .face_video import RAW_SOURCE_ERRORS, raw_root


STUDY = Path(__file__).resolve().parents[1]
STATE = STUDY / "common/outputs/face224_reruns"
INDEX_DIR = STUDY / "common/cache/face224_20frame"
GPU = None
INDEX = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment_plan():
    reg = STUDY / "exp2_face_pretrained_head32_regression"
    delta = STUDY / "exp6_face_pair_lab_delta"
    jobs = [{"key": "exp2_regression_24h", "family": "regression",
             "output": str(reg / "outputs/20frame_face224"), "hours": 24},
            {"key": "exp2_classification_24h", "family": "classification",
             "output": str(STUDY / "exp2_face_pretrained_head32_classification/outputs/face224"), "hours": 24}]
    for hours in (6, 12):
        jobs.append({"key": f"exp2_regression_{hours}h", "family": "regression",
                     "output": str(reg / f"outputs/ablations/lab_match_{hours}h_face224"), "hours": hours})
    for hours in (24, 6, 12):
        relative = "outputs/face224" if hours == 24 else f"outputs/ablations/lab_match_{hours}h_face224"
        jobs.append({"key": f"exp6_delta_{hours}h", "family": "delta", "output": str(delta / relative), "hours": hours})
    for grid in (4, 16):
        suffix = "" if grid == 4 else "_grid16"
        jobs.append({"key": f"exp8_grid{grid}", "family": "spectral", "grid": grid,
                     "output": str(STUDY / f"exp8_rgb_spectral_lab_regression/outputs{suffix}_face224")})
    return jobs


def load_reference(job, target):
    records = pd.read_csv(Path(job["output"]) / f"task_records/{target}.csv",
                          dtype={"hospital_id": str, "video_id": str})
    if set(records.split) != {"train", "val", "test"} or records.groupby("hospital_id").split.nunique().max() != 1:
        raise ValueError(f"Invalid saved patient split: {job['key']}/{target}")
    return records


def preflight(plan):
    """Freeze prepared native clinical inputs, never deleted 128 results."""
    from study.exp2_face_pretrained_head32_regression.config import WEIGHTS_DIR
    from study.exp2_face_pretrained_head32_regression.run_all import _validate_weights
    from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
    _validate_weights(WEIGHTS_DIR, ("efficientnet_b0",))
    index = FrameOffsetIndex.load(INDEX_DIR / "frame_offsets.npz")
    if set(index.video_formats) != {"ffv1"} or not _index_is_reusable(INDEX_DIR, index.video_ids, "20frame"):
        raise ValueError("A verified, reusable native-224 FFV1 index is required")
    fingerprints = {str(INDEX_DIR / "frame_offsets.npz"): sha256(INDEX_DIR / "frame_offsets.npz")}
    for job in plan:
        if job["family"] == "spectral":
            from study.exp8_rgb_spectral_lab_regression.spectral import checkpoint_sha256
            checkpoint_sha256()
            continue
        output = Path(job["output"])
        manifest = json.loads((output / "experiment_manifest.json").read_text())
        if manifest["frame_index_sha256"] != fingerprints[str(INDEX_DIR / "frame_offsets.npz")]:
            raise ValueError(f"Native frame index changed: {output}")
        if int(manifest["hours"]) != job["hours"]:
            raise ValueError(f"Matching protocol changed: {output}")
        prepare_experiment(job, index)
        for target in targets_for(job["family"]):
            path = output / f"task_records/{target}.csv"
            fingerprints[str(path)] = sha256(path)
        if job["family"] != "classification":
            fingerprints[str(output / "target_scalers.json")] = sha256(output / "target_scalers.json")
    return fingerprints


def prepare_experiment(job, index):
    from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler
    output = Path(job["output"])
    scalers = {} if job["family"] == "classification" else json.loads((output / "target_scalers.json").read_text())
    if job["family"] == "regression":
        scalers = {t: RobustTargetScaler(**v) for t, v in scalers["targets"].items()}
    records_by_target = {}
    for target in targets_for(job["family"]):
        records = load_reference(job, target)
        distance_columns = ["first_match_delta_h", "second_match_delta_h"] if job["family"] == "delta" else ["match_delta_h"]
        for column in distance_columns:
            distance = records[column].to_numpy(float)
            if not np.isfinite(distance).all() or np.any(distance < 0) or np.any(distance > job["hours"] + 1e-6):
                raise ValueError(f"Invalid saved matching interval: {job['key']}/{target}")
        columns = ["first_video_id", "second_video_id"] if job["family"] == "delta" else ["video_id"]
        for column in columns:
            if not set(records[column]).issubset(index.video_lookup):
                raise ValueError(f"Native source coverage changed: {job['key']}/{target}")
            if any(index.frame_range(v)[1] - index.frame_range(v)[0] != 20 for v in records[column]):
                raise ValueError("A saved video does not have twenty native frames")
        identity = "pair_id" if job["family"] == "delta" else "video_id"
        if records[identity].duplicated().any():
            raise ValueError("Duplicate clinical identities")
        if job["family"] == "regression":
            np.testing.assert_allclose(scalers[target].transform(records.raw_value),
                                       records.robust_scaled_raw_value, rtol=0, atol=1e-10)
        elif job["family"] == "delta":
            scaler = scalers[target]
            np.testing.assert_allclose((records.raw_delta - scaler["median"]) / scaler["iqr"],
                                       records.scaled_delta, rtol=0, atol=1e-10)
        records_by_target[target] = records
    if job["family"] == "delta":
        from study.exp6_face_pair_lab_delta.run_all import _validate_records
        _validate_records(records_by_target)
    return scalers


def validate_completed_job(job):
    """Reuse only a parent-confirmed, fully saved job with identical labels."""
    import torch
    run = Path(job["run_dir"])
    prediction_name = "pair_predictions.csv" if job["family"] == "delta" else "video_predictions.csv"
    if not all((run / name).is_file() for name in ("model.pt", "history.csv", "metrics.csv", prediction_name)):
        raise RuntimeError(f"An acknowledged completed job has missing artifacts: {run}")
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if checkpoint["target"] != job["target"] or not state:
        raise ValueError(f"Checkpoint identity mismatch: {run}")
    expected_scaler = job.get("scaler", job.get("target_scaler"))
    if hasattr(expected_scaler, "to_dict"):
        expected_scaler = expected_scaler.to_dict()
    if expected_scaler is not None and checkpoint["target_scaler"] != expected_scaler:
        raise ValueError(f"Checkpoint scaler mismatch: {run}")
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str})
    predicted = pd.read_csv(run / prediction_name, dtype={"hospital_id": str})
    identity = "pair_id" if job["family"] == "delta" else "video_id"
    truth = "raw_delta" if job["family"] == "delta" else ("binary_label" if job["family"] == "classification" else "raw_value")
    if predicted[identity].duplicated().any() or set(predicted[identity]) != set(records[identity]):
        raise ValueError(f"Prediction coverage mismatch: {run}")
    saved = predicted.set_index(identity).loc[records[identity]]
    if not saved.hospital_id.to_numpy().tolist() == records.hospital_id.to_numpy().tolist() or not np.array_equal(saved.split, records.split):
        raise ValueError(f"Prediction patient/split mismatch: {run}")
    np.testing.assert_allclose(saved["y_true" if job["family"] == "classification" else truth], records[truth], rtol=0, atol=1e-6)
    if not saved.frame_count.eq(20).all():
        raise ValueError(f"Prediction frame count mismatch: {run}")
    return True


def targets_for(family):
    if family == "delta":
        from study.exp6_face_pair_lab_delta.config import TARGETS
    elif family == "classification":
        from study.exp2_binary_classification_common.engine import TARGETS
    else:
        from study.exp2_face_pretrained_head32_regression.config import TARGETS
    return TARGETS


def preprocessing_ready():
    directory = raw_root() / "_face224_processing"
    with (directory / ".processing.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        protocol = json.loads((directory / "protocol.json").read_text())
        if protocol["geometry"].get("enable_alignment", True) or protocol["quality"]["confidence_threshold"] != .75:
            raise RuntimeError("The preprocessing job is not Kalman/no-alignment/confidence-0.75")
        table = pd.read_csv(directory / "index.csv")
        actual_sessions = sum(path.is_dir() for path in raw_root().glob("mirror*_data/patient_*"))
        if len(table) != actual_sessions or table.video_id.duplicated().any():
            raise RuntimeError("Full-video processing inventory is missing sessions")
        allowed = {"completed", "no_valid_frames", "missing_or_empty_video"}
        invalid_source = (table.status.eq("failed") & table.get("error", pd.Series("", index=table.index)).fillna("").str.startswith(RAW_SOURCE_ERRORS))
        if not (table.status.isin(allowed) | invalid_source).all():
            bad = table.loc[~(table.status.isin(allowed) | invalid_source), ["video_id", "status"]]
            raise RuntimeError(f"Video processing stopped with unfinished/failed sessions: {bad.to_dict('records')[:10]}")
        STATE.mkdir(parents=True, exist_ok=True)
        table.loc[~table.status.eq("completed")].to_csv(STATE / "preprocessing_exclusions.csv", index=False)
        return True


def init_worker(index_path, queue):
    global GPU, INDEX
    import torch
    from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
    GPU = int(queue.get())
    torch.cuda.set_device(GPU)
    torch.set_num_threads(1)
    INDEX = FrameOffsetIndex.load(index_path)


def train_job(job):
    if job["family"] == "regression":
        from study.exp2_face_pretrained_head32_regression import run_all as engine
        engine._WORKER_FRAME_INDEX, engine._WORKER_GPU_ID = INDEX, GPU
        result = engine._worker_train(job)
        engine._validate_saved_checkpoint(job["run_dir"], "efficientnet_b0", job["target"])
        return result["metrics"]
    if job["family"] == "delta":
        from study.exp6_face_pair_lab_delta import run_all as engine
        engine._FRAME_INDEX, engine._GPU_ID = INDEX, GPU
        return engine._worker_train(job)["metrics"]
    from study.exp2_binary_classification_common.engine import train_task
    return train_task("face_only", job["target"], GPU, job["seed"],
                      output_dir=Path(job["output"]), records_path=Path(job["records_path"]),
                      frame_index_path=INDEX_DIR / "frame_offsets.npz")


def run_experiment(experiment, index):
    import torch
    from study.exp2_face_pretrained_head32_regression import config as reg
    from study.exp6_face_pair_lab_delta import config as delta
    output = Path(experiment["output"])
    if (output / "COMPLETE").is_file():
        previous = json.loads((output / "experiment_manifest.json").read_text())
        if previous["frame_index_sha256"] != sha256(INDEX_DIR / "frame_offsets.npz"):
            raise RuntimeError("Completed result refers to a different frame index")
        print(f"[skip-complete] {experiment['key']}", flush=True)
        return
    scalers = prepare_experiment(experiment, index)
    jobs = []
    completed = {}
    saved_index = output / "run_index.csv"
    if saved_index.is_file():
        completed = pd.read_csv(saved_index).set_index("target").status.to_dict()
    reused_rows, reused_metrics = [], []
    for target in targets_for(experiment["family"]):
        run_dir = output / "runs" / (target if experiment["family"] == "delta" else f"efficientnet_b0/{target}")
        job = {**experiment, "target": target, "seed": reg.SEED,
               "records_path": str(output / f"task_records/{target}.csv"), "run_dir": str(run_dir),
               "max_batches": None}
        if experiment["family"] == "regression":
            job.update(architecture="efficientnet_b0", target_scaler=scalers[target], weights_dir=reg.WEIGHTS_DIR,
                       head_epochs=reg.HEAD_MAX_EPOCHS, finetune_epochs=reg.FINETUNE_MAX_EPOCHS,
                       head_patience=reg.HEAD_PATIENCE, finetune_patience=reg.FINETUNE_PATIENCE)
        elif experiment["family"] == "delta":
            job.update(scaler=scalers[target], seed=delta.SEED, head_epochs=delta.HEAD_MAX_EPOCHS,
                       finetune_epochs=delta.FINETUNE_MAX_EPOCHS, model_variant="shared", train_views=delta.VIEWS,
                       training_options={})
        if completed.get(target) in {"ok", "complete"} and validate_completed_job(job):
            reused_rows.append({"target": target, "architecture": "efficientnet_b0", "status": "ok"})
            reused_metrics.append(pd.read_csv(run_dir / "metrics.csv"))
            print(f"[reuse-job] {experiment['key']}/{target}", flush=True)
        else:
            jobs.append(job)
    workers = min(4, torch.cuda.device_count(), len(jobs))
    if jobs and workers < 1:
        raise RuntimeError("Four-GPU training requires CUDA")
    ctx = mp.get_context("spawn")
    rows, metrics, errors = reused_rows, reused_metrics, []
    with ctx.Manager() as manager:
        queue = manager.Queue()
        for gpu in range(workers):
            queue.put(gpu)
        print(f"[scheduler] {experiment['key']} tasks={len(jobs)} gpus={workers}", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, initializer=init_worker,
                                 initargs=(str(INDEX_DIR / "frame_offsets.npz"), queue)) as pool:
            futures = {pool.submit(train_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    metrics.append(pd.DataFrame(future.result()))
                    row = {"target": job["target"], "architecture": "efficientnet_b0", "status": "ok"}
                except Exception:
                    row = {"target": job["target"], "architecture": "efficientnet_b0", "status": "failed", "error": traceback.format_exc()}
                    errors.append(row)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
                if metrics:
                    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics_all.csv", index=False)
                print(f"[task-finished] {experiment['key']}/{job['target']} {row['status']}", flush=True)
    if errors:
        (output / "failures.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError(f"Training failed: {experiment['key']}")
    pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics_all.csv", index=False)
    histories = [pd.read_csv(path) for path in (output / "runs").rglob("history.csv")]
    pd.concat(histories, ignore_index=True).to_csv(output / "history_all.csv", index=False)
    if experiment["family"] == "regression":
        from study.exp2_face_pretrained_head32_regression.plot_results import main as plot
        plot(output)
    elif experiment["family"] == "delta":
        from study.exp6_face_pair_lab_delta.plot_results import plot_results
        plot_results(output)
    elif experiment["family"] == "classification":
        from .selected_5fold_plots import plot_fold_classification
        plot_fold_classification(output)
    if experiment["hours"] != 24:
        if experiment["family"] == "regression":
            from study.exp2_face_pretrained_head32_regression.plot_match_window_comparison import compare
            baseline_224 = STUDY / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
            compare(baseline_224, output, experiment["hours"])
        elif experiment["family"] == "delta":
            from study.exp6_face_pair_lab_delta.plot_match_window_comparison import plot_comparison as compare
            baseline_224 = STUDY / "exp6_face_pair_lab_delta/outputs/face224"
            compare(baseline_224, output, targets_for("delta"), experiment["hours"])
    (output / "COMPLETE").write_text("ok\n")


def run_spectral(job):
    env = os.environ.copy()
    env.update(EXP8_VARIANT="ntire2022", EXP8_GRID_SIZE=str(job["grid"]),
               EXP8_BASE_OUTPUT=str(STUDY / "exp2_face_pretrained_head32_regression/outputs/ablations/lab_match_12h_face224"),
               EXP8_INDEX_PATH=str(INDEX_DIR / "frame_offsets.npz"))
    if not (Path(job["output"]) / "COMPLETE").is_file():
        subprocess.run([sys.executable, "-u", "-m", "study.exp8_rgb_spectral_lab_regression.train"], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--run-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["HEALTHMIRROR_FACE_SOURCE"] = "face224"
    plan = experiment_plan()
    fingerprints = preflight(plan)
    if args.check_only:
        print(f"[preflight-ok] native jobs={len(plan)} frozen_inputs={len(fingerprints)}", flush=True)
        return
    from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
    if args.run_job:
        job = next(item for item in plan if item["key"] == args.run_job)
        if job["family"] == "spectral":
            run_spectral(job)
        else:
            run_experiment(job, FrameOffsetIndex.load(INDEX_DIR / "frame_offsets.npz"))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = {"schema_version": 2, "jobs": plan, "native_input_sha256": fingerprints}
        path = STATE / "plan.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("schema_version") == 2 and old != frozen:
                raise RuntimeError("Saved native input contracts changed")
        path.write_text(json.dumps(frozen, indent=2))
        (STATE / "COMPLETE").unlink(missing_ok=True)
        while not preprocessing_ready():
            print("[waiting] face224 preprocessing remains active", flush=True)
            time.sleep(args.poll_seconds)
        for job in plan:
            if preflight(plan) != fingerprints:
                raise RuntimeError("Native labels, scalers or frame index changed")
            log_dir = Path(job["output"].replace("/outputs", "/logs", 1))
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[experiment-start] {job['key']}", flush=True)
            with (log_dir / "run.log").open("a") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "study.common.rerun_face224", "--run-job", job["key"]],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if child.wait():
                    raise RuntimeError(f"Queue stopped: {job['key']}")
        (STATE / "COMPLETE").write_text("all native experiments and native comparisons completed\\n")
        print("[queue-complete] native-224 protocols completed", flush=True)


if __name__ == "__main__":
    main()
