"""Train histories first, then independent face regressors on OOF residuals."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_face_history_head32_regression.history_data import HistoryFeatureStore
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from study.exp2_face_history_head32_regression.source_data import _sha256
from study.exp2_face_pretrained_head32_regression import train as face_train
from study.exp2_face_pretrained_head32_regression.config import SCORE_DEFINITIONS, SEED, WEIGHTS_DIR
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.run_all import _set_seed
from study.exp2_history_only_head32_regression.train import train_task as train_history
from study.exp2_lab_multimodal.config import LAB_CSV

from .data import HERE, REFERENCE, TASKS, ResidualScaler, crossfit_splits, prepare, residual_records, rescale_history


GPU = None
INDEX = None


def release_gpu_cache():
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execution_model(model, architecture, target, stage, device):
    if device.type != "cuda" or not face_train.TORCH_COMPILE_ENABLED:
        return model, "eager"
    compiled = torch.compile(model, dynamic=True, options={"triton.cudagraphs": False})
    print(f"[compile-enabled] task={target} stage={stage} mode=default dynamic=True cudagraphs=False evaluation=eager", flush=True)
    return compiled, "inductor:default_dynamic_no_cudagraphs"


def clear_training_results(output):
    """Keep prepared labels/histories/index references; overwrite generated fits."""
    for name in ("runs", "residual_records", "figures", "tables"):
        path = output / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("history_run_index.csv", "face_run_index.csv", "COMPLETED.json", "REPORT.md"):
        (output / name).unlink(missing_ok=True)


def initialize(queue, index_path):
    global GPU, INDEX
    GPU = int(queue.get())
    torch.cuda.set_device(GPU)
    torch.set_num_threads(1)
    INDEX = FrameOffsetIndex.load(index_path)
    print(f"[worker-ready] pid={__import__('os').getpid()} gpu={GPU}", flush=True)


def load_inputs(output, target):
    records = pd.read_csv(output / f"task_records/{target}.csv", dtype={"hospital_id": str})
    store = HistoryFeatureStore.load(output / f"history_records/{target}.npz")
    scaler = RobustTargetScaler(**json.loads((output / "target_scalers.json").read_text())[target])
    records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
    return records, store, scaler


def history_job(job):
    output, target = Path(job["output"]), job["target"]
    records, store, scaler = load_inputs(output, target)
    seed = job["seed"] + TASKS.index(target)*100
    options = {"head_epochs": job["head_epochs"], "finetune_epochs": job["finetune_epochs"],
               "head_patience": 10, "finetune_patience": 12}
    device = torch.device(f"cuda:{GPU}")
    _set_seed(seed)
    full_path = output / f"runs/{target}/history_full"
    print(f"[history-full-start] task={target} gpu={GPU}", flush=True)
    train_history(target, records, store, scaler, full_path, device, seed, **options)
    full = pd.read_csv(full_path / "video_predictions.csv", dtype={"hospital_id": str})
    baseline = full.loc[full.split.ne("train"), ["hospital_id", "video_id", "y_pred"]].rename(columns={"y_pred": "history_pred_raw"})
    baseline["prediction_source"] = "outer_train_fit"
    raw_history = (pd.read_csv(output / f"history_records/{target}.csv", usecols=["history_value"])["history_value"].to_numpy(np.float64)
                   if len(store.features) else np.empty(0, dtype=np.float64))
    oof_parts, membership = [], []
    for fold, fold_records in enumerate(crossfit_splits(records, seed)):
        fold_seed = seed+fold+1
        _set_seed(fold_seed)
        fold_scaler = fit_robust_target_scaler(target, fold_records, scaler.unit)
        fold_records["robust_scaled_raw_value"] = fold_scaler.transform(fold_records.raw_value)
        fold_store = rescale_history(store, fold_scaler, raw_history)
        path = output / f"runs/{target}/history_oof/fold{fold+1}"
        print(f"[history-oof-start] task={target} fold={fold+1}/3 gpu={GPU}", flush=True)
        path.mkdir(parents=True, exist_ok=True)
        fold_records.to_csv(path / "patient_split.csv", index=False)
        train_history(target, fold_records, fold_store, fold_scaler, path, device, fold_seed, **options)
        predictions = pd.read_csv(path / "video_predictions.csv", dtype={"hospital_id": str})
        held_out = predictions.loc[predictions.split.eq("test"), ["hospital_id", "video_id", "y_pred"]].rename(columns={"y_pred": "history_pred_raw"})
        held_out["prediction_source"] = "patient_oof"
        oof_parts.append(held_out)
        membership.append(fold_records[["hospital_id", "video_id", "split"]].assign(fold=fold+1))
    baseline = pd.concat([baseline, *oof_parts], ignore_index=True)
    residual = residual_records(records, baseline, scaler)
    (output / "residual_records").mkdir(exist_ok=True)
    residual.to_csv(output / f"residual_records/{target}.csv", index=False)
    baseline.to_csv(output / f"runs/{target}/history_baseline_predictions.csv", index=False)
    pd.concat(membership, ignore_index=True).to_csv(output / f"runs/{target}/crossfit_membership.csv", index=False)
    print(f"[history-done] task={target} OOF/train={residual.split.eq('train').sum()} residual_mean={residual.loc[residual.split.eq('train'), 'raw_value'].mean():.4f}", flush=True)
    return target


def composed_metrics(evaluation, dataset, split, target, scaler, original):
    _, frames, videos = original(evaluation, dataset, split, target, scaler)
    context = dataset.video_records[["video_id", "original_raw_value", "history_pred_raw", "original_score_threshold"]]
    merged = videos.merge(context, on="video_id", how="left", validate="one_to_one")
    metrics = face_train._regression_metrics(merged.original_raw_value, merged.history_pred_raw+merged.y_pred,
                                            merged.original_score_threshold, SCORE_DEFINITIONS[target]["direction"])
    return metrics, frames, videos


def _face_job_impl(job):
    output, target = Path(job["output"]), job["target"]
    _, _, lab_scaler = load_inputs(output, target)
    records = pd.read_csv(output / f"residual_records/{target}.csv", dtype={"hospital_id": str})
    scaler = ResidualScaler(target, lab_scaler.unit, lab_scaler.iqr, lab_scaler.to_dict())
    records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
    _set_seed(job["seed"]+TASKS.index(target)*100)
    path = output / f"runs/{target}/face_residual"
    original_metrics = face_train._video_metrics
    original_compile = face_train.TORCH_COMPILE_ENABLED
    original_execution = face_train._execution_model
    original_evaluate = face_train._evaluate
    face_train._video_metrics = lambda *args: composed_metrics(*args, original=original_metrics)
    face_train.TORCH_COMPILE_ENABLED = not job["smoke"]
    face_train._execution_model = execution_model
    face_train._evaluate = lambda model, *args, **kwargs: original_evaluate(getattr(model, "_orig_mod", model), *args, **kwargs)
    try:
        face_train.train_task("efficientnet_b0", target, INDEX, records, scaler, WEIGHTS_DIR, str(path),
                              head_epochs=job["head_epochs"], finetune_epochs=job["finetune_epochs"],
                              max_batches=1 if job["smoke"] else None,
                              train_batch_policy="distinct_lab_views", loss_level="frame")
    finally:
        face_train._video_metrics = original_metrics
        face_train.TORCH_COMPILE_ENABLED = original_compile
        face_train._execution_model = original_execution
        face_train._evaluate = original_evaluate
    (path / "video_predictions.csv").replace(path / "residual_video_predictions.csv")
    (path / "frame_predictions.npz").replace(path / "residual_frame_predictions.npz")
    residual_pred = pd.read_csv(path / "residual_video_predictions.csv", dtype={"hospital_id": str})
    combined = residual_pred.merge(records[["video_id", "original_raw_value", "original_score_threshold", "history_pred_raw", "prediction_source"]],
                                   on="video_id", how="left", validate="one_to_one")
    combined["predicted_residual_raw"] = combined.y_pred
    combined["true_residual_raw"] = combined.y_true
    combined["y_pred"] = combined.history_pred_raw + combined.predicted_residual_raw
    combined["y_true"] = combined.original_raw_value
    combined["raw_value"] = combined.original_raw_value
    combined["score_threshold"] = combined.original_score_threshold
    combined["y_pred_scaled"] = lab_scaler.transform(combined.y_pred)
    combined["y_true_scaled"] = lab_scaler.transform(combined.y_true)
    combined["residual"] = combined.y_pred-combined.y_true
    combined["architecture"] = "history_plus_face_residual"
    combined.to_csv(path / "video_predictions.csv", index=False)
    metric_rows = []
    for split, group in combined.groupby("split"):
        for name, prediction in (("history_baseline", group.history_pred_raw), ("history_plus_face_residual", group.y_pred)):
            metrics = face_train._regression_metrics(group.y_true, prediction, group.score_threshold, SCORE_DEFINITIONS[target]["direction"])
            metric_rows.append({"target": target, "split": split, "architecture": name, **metrics})
    pd.DataFrame(metric_rows).to_csv(path / "comparison_metrics.csv", index=False)
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    checkpoint.update({"task_type": "history_conditioned_face_residual_regression",
                       "model_output": "raw residual / outer-train raw-label IQR",
                       "reported_predictions": "frozen history prediction + mean of twenty original-frame residual predictions",
                       "frame_policy": "20 deterministic non-adjacent native224 FFV1 frames",
                       "execution": {"backend": "inductor", "mode": "default", "dynamic_batches": True,
                                     "cudagraphs": False, "evaluation": "eager"},
                       "train_residual_policy": "three-fold patient OOF; inner train-only validation and scaler",
                       "lab_target_scaler": lab_scaler.to_dict()})
    torch.save(checkpoint, path / "model.pt")
    history = torch.load(output / f"runs/{target}/history_full/model.pt", map_location="cpu", weights_only=True)
    torch.save({"target": target, "task_type": "raw_lab_value_history_plus_face_residual",
                "history_state_dict": history["model_state_dict"], "face_state_dict": checkpoint["model_state_dict"],
                "lab_target_scaler": lab_scaler.to_dict(), "residual_scaler": scaler.to_dict(),
                "history_policy": "strictly prior same-analyte, same unique admission; missing history retained",
                "frame_aggregation": "mean of twenty original-frame predictions at evaluation",
                "head_hidden_features": 32}, output / f"runs/{target}/model.pt")
    # Reload both components so a completed job cannot silently lack a usable checkpoint.
    from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model
    from study.exp2_history_only_head32_regression.models import HistoryOnlyRegressor
    saved = torch.load(output / f"runs/{target}/model.pt", map_location="cpu", weights_only=True)
    HistoryOnlyRegressor().load_state_dict(saved["history_state_dict"], strict=True)
    model, _, _ = build_pretrained_model("efficientnet_b0", WEIGHTS_DIR)
    model.load_state_dict(saved["face_state_dict"], strict=True)
    del model
    return target


def face_job(job):
    release_gpu_cache()
    try:
        return _face_job_impl(job)
    finally:
        release_gpu_cache()


def run_stage(stage, jobs, devices, index_path, output):
    context = mp.get_context("spawn")
    queue = context.Queue()
    for device in devices:
        queue.put(device)
    statuses = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context, initializer=initialize,
                             initargs=(queue, str(index_path))) as pool:
        futures = {pool.submit(history_job if stage == "history" else face_job, job): job["target"] for job in jobs}
        for future in as_completed(futures):
            target = futures[future]
            try:
                future.result()
                statuses.append({"target": target, "stage": stage, "status": "completed"})
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                statuses.append({"target": target, "stage": stage, "status": "failed", "error": error})
            pd.DataFrame(statuses).to_csv(output / f"{stage}_run_index.csv", index=False)
            print(f"[progress] stage={stage} complete={len(statuses)}/{len(jobs)} task={target} status={statuses[-1]['status']}", flush=True)
    if any(item["status"] != "completed" for item in statuses):
        raise RuntimeError(f"{stage} jobs failed; later stages will not launch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(TASKS))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-prepared", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.overwrite and not args.train_prepared:
        raise ValueError("--overwrite requires --train-prepared to retain the exact input data")
    targets = tuple(args.targets.split(","))
    if not set(targets).issubset(TASKS) or len(set(targets)) != len(targets):
        raise ValueError("Unknown or repeated targets")
    output = args.output_dir.resolve()
    if args.smoke_test and output == HERE / "outputs":
        output = HERE / "outputs_smoke"
    if (output / "protocol.json").exists() and not args.train_prepared:
        raise FileExistsError(f"Output already exists; choose a new --output-dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.train_prepared:
        previous = json.loads((output / "protocol.json").read_text())
        lab_sha = _sha256(LAB_CSV)
        if (args.prepare_only or previous["lab_sha256"] != lab_sha or previous["targets"] != list(targets)
                or previous["seed"] != args.seed or previous["smoke_test"] != args.smoke_test
                or ((output / "history_run_index.csv").exists() and not args.overwrite)):
            raise ValueError("Prepared protocol changed or training has already started")
        for target in targets:
            if previous["reference_record_hashes"][target] != _sha256(REFERENCE / f"task_records/{target}.csv"):
                raise ValueError("Reference patient split changed after preparation")
    else:
        lab_sha = prepare(output, targets, LAB_CSV, args.seed, args.smoke_test)
    protocol = json.loads((REFERENCE / "direct_field_extension_protocol.json").read_text())
    index_path = Path(protocol["frame_index"])
    index = FrameOffsetIndex.load(index_path)
    if (set(index.video_formats) != {"ffv1"} or not _index_is_reusable(index_path.parent, index.video_ids, "20frame")):
        raise RuntimeError("The shared native224 frame index is not reusable")
    for target in targets:
        records = pd.read_csv(output / f"task_records/{target}.csv")
        for video in records.video_id:
            start, end = index.frame_range(video)
            if end-start != 20:
                raise RuntimeError(f"Invalid frame count: {video}")
    if args.overwrite:
        clear_training_results(output)
    manifest = {"targets": targets, "lab_sha256": lab_sha, "seed": args.seed,
                "outer_split": "exact per-target current ten-task face-only primary split",
                "reference": str(REFERENCE), "reference_record_hashes": {t: _sha256(REFERENCE / f"task_records/{t}.csv") for t in targets},
                "frame_index": str(index_path), "frame_index_sha256": _sha256(index_path),
                "history_crossfit_folds": 3, "inner_validation_fraction": .20,
                "history_encoder_parameters": 352, "history_head_parameters": 641,
                "head_hidden_features": 32, "image_size": 224, "frames": 20,
                "views": ["original", "hflip", "center_crop", "brightness", "contrast"],
                "history_lr": [2e-4, 1e-5], "face_lr": [2e-4, 1e-5], "epochs": [40, 60],
                "patience": [10, 12], "optimizer": "AdamW", "weight_decay": 1e-4,
                "scheduler": "per-stage cosine to 1e-6", "loss": "unweighted SmoothL1(beta=0.5)",
                "loss_level": "frame", "batch": "12 distinct clinical events, 20 frames, one view each = 240 inputs",
                "reporting": "video-level raw units; training baseline OOF, val/test outer-train fit",
                "history_time_cutoff": "strictly before current label, not video time; not a prospective real-time protocol",
                "matching": "24h nearest in same hospitalization, unchanged audited source whitelist",
                "history_training": "same-analyte MLP; both LR stages train all history parameters",
                "gpu_stage_barrier": "all full/OOF histories finish before any face training",
                "face_execution": {"backend": "inductor", "mode": "default", "dynamic_batches": True,
                                   "cudagraphs": False, "evaluation": "eager", "cache_reset_per_task": True,
                                   "reason": "Previous reduce-overhead CUDA Graph private pools grew to 13.39 GiB and caused repeated worker OOM"},
                "restart": "all history/OOF/face models from scratch, prepared data unchanged" if args.overwrite else None,
                "smoke_test": args.smoke_test}
    if args.smoke_test:
        manifest["epochs"] = [1, 1]
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.prepare_only:
        return
    devices = [int(value) for value in args.devices.split(",")]
    if not torch.cuda.is_available() or len(set(devices)) != len(devices) or any(g < 0 or g >= torch.cuda.device_count() for g in devices):
        raise ValueError("Valid distinct CUDA devices are required")
    jobs = [{"output": str(output), "target": target, "seed": args.seed, "smoke": args.smoke_test,
             "head_epochs": 1 if args.smoke_test else 40, "finetune_epochs": 1 if args.smoke_test else 60} for target in targets]
    run_stage("history", jobs, devices, index_path, output)
    run_stage("face", jobs, devices, index_path, output)
    from .plots import plot_results
    plot_results(output, targets)
    (output / "COMPLETED.json").write_text(json.dumps({"targets": targets, "figures": str(output / "figures"),
                                                     "completed_utc": pd.Timestamp.now(tz="UTC").isoformat()}, indent=2)+"\n")


if __name__ == "__main__":
    main()
