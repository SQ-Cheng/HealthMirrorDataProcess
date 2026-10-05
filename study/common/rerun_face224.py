"""Wait for verified face224 preprocessing, then rerun controlled experiments."""

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
    jobs = [{"key": "exp2_regression_24h", "family": "regression", "baseline": str(reg / "outputs/20frame"),
             "output": str(reg / "outputs/20frame_face224"), "hours": 24},
            {"key": "exp2_classification_24h", "family": "classification",
             "baseline": str(STUDY / "exp2_face_pretrained_head32_classification/outputs"),
             "output": str(STUDY / "exp2_face_pretrained_head32_classification/outputs/face224"), "hours": 24}]
    for hours in (6, 12):
        jobs.append({"key": f"exp2_regression_{hours}h", "family": "regression",
                     "baseline": str(reg / f"outputs/ablations/lab_match_{hours}h"),
                     "output": str(reg / f"outputs/ablations/lab_match_{hours}h_face224"), "hours": hours})
    for hours in (24, 6, 12):
        relative = "outputs" if hours == 24 else f"outputs/ablations/lab_match_{hours}h"
        jobs.append({"key": f"exp6_delta_{hours}h", "family": "delta", "baseline": str(delta / relative),
                     "output": str(delta / ("outputs/face224" if hours == 24 else relative + "_face224")), "hours": hours})
    for grid in (4, 16):
        suffix = "" if grid == 4 else "_grid16"
        jobs.append({"key": f"exp8_grid{grid}", "family": "spectral", "grid": grid,
                     "baseline": str(STUDY / f"exp8_rgb_spectral_lab_regression/outputs{suffix}"),
                     "output": str(STUDY / f"exp8_rgb_spectral_lab_regression/outputs{suffix}_face224")})
    return jobs


def targets_for(family):
    if family == "delta":
        from study.exp6_face_pair_lab_delta.config import TARGETS
    elif family == "classification":
        from study.exp2_binary_classification_common.engine import TARGETS
    else:
        from study.exp2_face_pretrained_head32_regression.config import TARGETS
    return TARGETS


def load_reference(job, target):
    if job["family"] == "classification":
        from study.exp2_binary_classification_common.engine import load_task
        records, _ = load_task(target)
    else:
        records = pd.read_csv(Path(job["baseline"]) / f"task_records/{target}.csv",
                              dtype={"hospital_id": str, "video_id": str})
    if set(records.split) != {"train", "val", "test"} or records.groupby("hospital_id").split.nunique().max() != 1:
        raise ValueError(f"Invalid reference patient split: {job['key']}/{target}")
    return records


def preflight(plan):
    """Freeze existing labels/splits/results before waiting, without reading videos."""
    from study.exp2_face_pretrained_head32_regression.config import WEIGHTS_DIR
    from study.exp2_face_pretrained_head32_regression.run_all import _validate_weights
    _validate_weights(WEIGHTS_DIR, ("efficientnet_b0",))
    fingerprints = {}
    for job in plan:
        baseline = Path(job["baseline"])
        for name in ("metrics_all.csv", "history_all.csv") if job["family"] == "spectral" else ("run_index.csv", "metrics_all.csv"):
            path = baseline / name
            if not path.is_file():
                raise FileNotFoundError(path)
            fingerprints[str(path)] = sha256(path)
        if job["family"] == "spectral":
            from study.exp8_rgb_spectral_lab_regression.spectral import checkpoint_sha256
            checkpoint_sha256()
            from study.exp8_rgb_spectral_lab_regression.config import TARGETS
            for target in TARGETS:
                path = baseline / f"runs/{target}/predictions.csv"
                fingerprints[str(path)] = sha256(path)
            continue
        statuses = pd.read_csv(baseline / "run_index.csv")
        if not statuses.status.isin(["ok", "complete"]).all() or set(statuses.target) != set(targets_for(job["family"])):
            raise RuntimeError(f"Incomplete baseline: {baseline}")
        for target in targets_for(job["family"]):
            records = load_reference(job, target)
            if job["family"] == "classification":
                from study.exp2_binary_classification_common.engine import PREPARED_DIR, REFERENCE_DIR
                path = (PREPARED_DIR if target == "total_bilirubin_high" else REFERENCE_DIR) / f"task_records/{target}.csv"
                prediction = pd.read_csv(baseline / f"runs/efficientnet_b0/{target}/video_predictions.csv", dtype={"hospital_id": str})
                identities = ["hospital_id", "video_id", "split", "binary_label"]
                pd.testing.assert_frame_equal(records[identities].sort_values("video_id").reset_index(drop=True),
                                              prediction[identities].sort_values("video_id").reset_index(drop=True), check_dtype=False)
            else:
                path = baseline / f"task_records/{target}.csv"
            fingerprints[str(path)] = sha256(path)
            prediction_path = (baseline / f"runs/{target}/pair_predictions.csv"
                               if job["family"] == "delta" else baseline / f"runs/efficientnet_b0/{target}/video_predictions.csv")
            fingerprints[str(prediction_path)] = sha256(prediction_path)
        if job["family"] != "classification":
            path = baseline / "target_scalers.json"
            fingerprints[str(path)] = sha256(path)
    return fingerprints


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


def video_rows(plan):
    identities = {}
    for job in plan:
        if job["family"] == "spectral":
            continue
        for target in targets_for(job["family"]):
            records = load_reference(job, target)
            if job["family"] == "delta":
                video_columns = ("first_video_id", "second_video_id")
            else:
                video_columns = ("video_id",)
            for column in video_columns:
                for video_id, patient in zip(records[column], records.hospital_id):
                    if video_id in identities and identities[video_id] != str(patient):
                        raise ValueError(f"Inconsistent source patient identity: {video_id}")
                    identities[video_id] = str(patient)
    rows = []
    for video_id in sorted(identities):
        mirror, patient = video_id.split("_patient_")
        rows.append({"video_id": video_id, "mirror": mirror, "lab_patient_id": int(patient),
                     "hospital_id": identities[video_id]})
    return pd.DataFrame(rows)


def prepare_experiment(job, index):
    from study.exp2_face_pretrained_head32_regression.scaling import fit_robust_target_scaler, write_target_scalers
    from study.exp2_face_pretrained_head32_regression.config import SCORE_DEFINITIONS
    from study.exp6_face_pair_lab_delta.run_all import _validate_records
    output = Path(job["output"])
    task_dir = output / "task_records"
    task_dir.mkdir(parents=True, exist_ok=True)
    records_by_target, scalers, summaries, excluded = {}, {}, [], []
    usable = set(index.video_lookup)
    index_audit = pd.read_csv(INDEX_DIR / "video_frame_summary.csv")
    frame_reasons = dict(zip(index_audit.video_id, index_audit.get("reason", index_audit.status)))
    for target in targets_for(job["family"]):
        original = load_reference(job, target)
        mask = (original.first_video_id.isin(usable) & original.second_video_id.isin(usable)
                if job["family"] == "delta" else original.video_id.isin(usable))
        records = original.loc[mask].reset_index(drop=True).copy()
        removed = original.loc[~mask].assign(target=target, exclusion_reason="no_20_nonadjacent_accepted_face224_frames")
        if job["family"] == "delta":
            removed["first_video_exclusion"] = removed.first_video_id.map(frame_reasons)
            removed["second_video_exclusion"] = removed.second_video_id.map(frame_reasons)
        else:
            removed["video_exclusion"] = removed.video_id.map(frame_reasons)
        excluded.append(removed)
        if set(records.split) != {"train", "val", "test"}:
            raise RuntimeError(f"224 frame exclusion emptied a split: {target}")
        if records.groupby("hospital_id").split.nunique().max() != 1:
            raise RuntimeError(f"Patient leakage: {target}")
        train = records.loc[records.split.eq("train")]
        if job["family"] == "delta":
            value = train.raw_delta.to_numpy(np.float64)
            q1, med, q3 = np.quantile(value, [.25, .5, .75])
            if q3 <= q1:
                raise RuntimeError(f"Zero train-only delta IQR: {target}")
            scaler = {"target": target, "unit": records.unit.iloc[0], "fit_split": "train", "median": float(med), "iqr": float(q3 - q1)}
            records["scaled_delta"] = (records.raw_delta - med) / (q3 - q1)
        elif job["family"] == "regression":
            scaler = fit_robust_target_scaler(target, records, SCORE_DEFINITIONS[target]["unit"])
            records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
        else:
            for split, subset in records.groupby("split"):
                if set(subset.binary_label.astype(int)) != {0, 1}:
                    raise RuntimeError(f"Single-class retained {split}: {target}")
            scaler = None
        if scaler is not None:
            scalers[target] = scaler
        records.to_csv(task_dir / f"{target}.csv", index=False)
        records_by_target[target] = records
        for split, subset in records.groupby("split"):
            summaries.append({"target": target, "split": split, "videos_or_pairs": len(subset),
                              "patients": subset.hospital_id.nunique(),
                              "reference_rows": int(original.split.eq(split).sum()),
                              "removed_rows": int(original.split.eq(split).sum() - len(subset))})
    if job["family"] == "delta":
        _validate_records(records_by_target)
        (output / "target_scalers.json").write_text(json.dumps(scalers, indent=2))
    elif job["family"] == "regression":
        write_target_scalers(scalers, output / "target_scalers.json")
        source = Path(job["baseline"]) / "source_data"
        # Keep Exp8's clinical source validation independent of processed length.
        import shutil
        shutil.copytree(source, output / "source_data", dirs_exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "cohort_counts.csv", index=False)
    pd.concat(excluded, ignore_index=True).to_csv(output / "excluded_records.csv", index=False)
    manifest = {**job, "targets": list(records_by_target), "frame_index": str(INDEX_DIR / "frame_offsets.npz"),
                "frame_index_sha256": sha256(INDEX_DIR / "frame_offsets.npz"),
                "difference": "128 MJPEG crops replaced by native 224 lossless Kalman crops; no alignment",
                "split_policy": "exact saved per-target patient assignments for all retained records; no new split search",
                "matching_policy": "exact saved original-video/lab matching and labels; never recompute clinical times from shortened face video",
                "scaling": "refit median/IQR on retained training rows only",
                "training": "same model, head32, 20 nonadjacent frames, 5 training views, original-only eval, job seeds and native training defaults",
                "comparison": "full cohorts and common test identities reported separately"}
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return scalers


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
        jobs.append(job)
    workers = min(4, torch.cuda.device_count(), len(jobs))
    if workers < 1:
        raise RuntimeError("Four-GPU training requires CUDA")
    ctx = mp.get_context("spawn")
    rows, metrics, errors = [], [], []
    with ctx.Manager() as manager:
        queue = manager.Queue()
        for gpu in range(workers):
            queue.put(gpu)
        print(f"[scheduler] {experiment['key']} tasks={len(jobs)} gpus={workers}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
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
    histories = [pd.read_csv(path) for path in (output / "runs").rglob("history.csv")]
    pd.concat(histories, ignore_index=True).to_csv(output / "history_all.csv", index=False)
    if experiment["family"] == "regression":
        from study.exp2_face_pretrained_head32_regression.plot_results import main as plot
        plot(output)
    elif experiment["family"] == "delta":
        from study.exp6_face_pair_lab_delta.plot_results import plot_results
        plot_results(output)
    from .plot_face224_comparison import plot_comparison
    plot_comparison(experiment)
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
               EXP8_BASE_OUTPUT=str(STUDY / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"),
               EXP8_INDEX_PATH=str(INDEX_DIR / "frame_offsets.npz"))
    if not (Path(job["output"]) / "COMPLETE").is_file():
        subprocess.run([sys.executable, "-u", "-m", "study.exp8_rgb_spectral_lab_regression.train"], env=env, check=True)
    from .plot_face224_comparison import plot_comparison
    plot_comparison(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--run-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["HEALTHMIRROR_FACE_SOURCE"] = "face224"
    plan = experiment_plan()
    print("[preflight] checking frozen reference labels, splits, results and local weights", flush=True)
    fingerprints = preflight(plan)
    if args.check_only:
        print(json.dumps(plan, indent=2))
        print(f"[preflight-ok] jobs={len(plan)} baseline_files={len(fingerprints)}")
        return
    if args.run_job:
        job = next(item for item in plan if item["key"] == args.run_job)
        if job["family"] == "spectral":
            run_spectral(job)
        else:
            from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
            run_experiment(job, FrameOffsetIndex.load(INDEX_DIR / "frame_offsets.npz"))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = STATE / "plan.json"
        frozen = {"jobs": plan, "baseline_sha256": fingerprints}
        if plan_path.exists() and json.loads(plan_path.read_text()) != frozen:
            raise RuntimeError("Frozen baseline results changed; review the queue plan before proceeding")
        plan_path.write_text(json.dumps(frozen, indent=2))
        while not preprocessing_ready():
            print("[waiting] face224 processing remains active; training not started", flush=True)
            time.sleep(args.poll_seconds)
        if preflight(plan) != fingerprints:
            raise RuntimeError("Baseline artifacts changed while waiting")
        from study.exp2_face_pretrained_head32_regression.frame_index import build_or_reuse_frame_index
        print("[index-validation] resolving raw sessions and building the shared native 224 index", flush=True)
        index = build_or_reuse_frame_index(video_rows(plan), INDEX_DIR)
        if not len(index.video_ids) or set(index.video_formats) != {"ffv1"}:
            raise RuntimeError("The retraining index must contain only validated 224 FFV1 videos")
        print(f"[preprocessing-ready] usable_videos={len(index.video_ids)} frames={len(index.starts)}", flush=True)
        for job in plan:
            if preflight(plan) != fingerprints:
                raise RuntimeError("Frozen baseline artifacts changed between experiments")
            log_dir = Path(job["output"].replace("/outputs", "/logs", 1))
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[experiment-start] {job['key']} log={log_dir / 'run.log'}", flush=True)
            with (log_dir / "run.log").open("a") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "study.common.rerun_face224", "--run-job", job["key"]],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if child.wait():
                    raise RuntimeError(f"Queue stopped: {job['key']} failed; see {log_dir / 'run.log'}")
        comparisons = [pd.read_csv(Path(job["output"]) / "face224_comparison.csv").assign(experiment=job["key"]) for job in plan]
        pd.concat(comparisons, ignore_index=True).to_csv(STATE / "comparison_all.csv", index=False)
        report = ["# Native 224 crop reruns", "", "Original results are retained. Clinical labels and saved patient splits are reused.",
                  "Full-test cohorts can differ; common-test plots compare identical held-out identities and labels.",
                  "Training cohorts may shrink, and scalers are fitted on retained train rows only; differences are not attributable solely to resolution.", ""]
        for job in plan:
            report.append(f"- {job['key']}: {job['output']}/figures/face224_vs_legacy_common_test.png")
        (STATE / "REPORT.md").write_text("\n".join(report) + "\n")
        (STATE / "COMPLETE").write_text("all requested experiments and comparisons completed\n")
        print("[queue-complete] all 224 experiments and paired comparisons generated", flush=True)


if __name__ == "__main__":
    main()
