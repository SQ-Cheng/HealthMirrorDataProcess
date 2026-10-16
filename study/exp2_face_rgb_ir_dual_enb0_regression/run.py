"""Eight-GPU queue for ten independent dual-encoder two-stage regressors."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.common.run_video_loss_12h import sha256, Tee
from study.exp2_face_pretrained_head32_regression import train as primary
from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model as rgb_model, WEIGHT_FILES
from study.exp2_face_pretrained_head32_regression.run_all import _set_seed
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from study.exp2_face_rgb_ir_head32_regression.index import build_indexes, load_indexes
from . import config
from .engine import adapters, evaluate


RGB = IR = DEVICE = None


class TrainingTee(Tee):
    @property
    def encoding(self):
        # Triton decodes GPU hardware queries using sys.stdout.encoding.
        return getattr(self.stream, "encoding", None) or "utf-8"


def records_at(path):
    return pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")


def prepare():
    paired = config.paired
    source = paired.OUTPUT
    old = json.loads((source / "protocol.json").read_text())
    if not (source / "COMPLETED.json").is_file():
        raise RuntimeError("Four-channel RGB/IR comparator is incomplete")
    lab_hash = sha256(paired.ROOT / "merged_lab_tests.csv")
    if old["smoke_test"] or old["lab_table_sha256"] != lab_hash or tuple(old["targets"]) != tuple(config.TARGETS):
        raise RuntimeError("Paired source must use the latest full ten-task lab table")
    if old["paired_table_sha256"] != sha256(paired.PAIRED / "tables/selected20_pairs.csv"):
        raise RuntimeError("Synchronous frame selection changed")
    summary = json.loads((paired.PAIRED / "summary.json").read_text())
    if not summary["complete_for_candidate_population"] or summary["quality_unknown_candidate_pairs"]:
        raise RuntimeError("Incomplete IR audit")
    main = json.loads((paired.REFERENCE / "main_protocol.json").read_text())
    extension = json.loads((paired.REFERENCE / "direct_field_extension_protocol.json").read_text())
    main_hashes = {**main["source_records_sha256"], **extension["record_hashes"][str(paired.REFERENCE)]}
    if main["lab_table_sha256"] != lab_hash or extension["lab_table_sha256"] != lab_hash:
        raise RuntimeError("RGB comparator uses a different lab table")
    frames = {target: records_at(source / f"task_records/{target}.csv") for target in config.TARGETS}
    scalers = json.loads((source / "target_scalers.json").read_text())
    rgb_path, ir_path = build_indexes(video for frame in frames.values() for video in frame.video_id)
    rgb, _ = load_indexes(rgb_path, ir_path)
    hashes, rgb_checkpoints, control_predictions, counts = {}, {}, {}, []
    for target, frame in frames.items():
        original_path = paired.REFERENCE / f"task_records/{target}.csv"
        if sha256(original_path) != main_hashes[target] or old["reference_record_hashes"][target] != main_hashes[target]:
            raise RuntimeError("Main/paired task records changed")
        original = records_at(original_path).set_index("video_id").loc[frame.video_id].reset_index()
        if (frame.video_id.duplicated().any() or frame.groupby("hospital_id").split.nunique().gt(1).any()
                or set(frame.split) != {"train", "val", "test"} or frame.clinical_event_id.isna().any()
                or not frame.match_delta_h.between(0, 24+1e-9).all()):
            raise ValueError("Invalid paired split or labels")
        for column in ("hospital_id", "split", "clinical_event_id"):
            np.testing.assert_array_equal(frame[column], original[column])
        np.testing.assert_allclose(frame.raw_value, original.raw_value, rtol=1e-12, atol=1e-12)
        scaler = RobustTargetScaler(**scalers[target])
        if scaler.to_dict() != fit_robust_target_scaler(target, frame, scaler.unit).to_dict():
            raise ValueError("Paired scaler is not train-only")
        np.testing.assert_allclose(frame.robust_scaled_raw_value, scaler.transform(frame.raw_value), rtol=1e-6, atol=1e-6)
        if not np.isfinite(frame[["raw_value", "robust_scaled_raw_value"]]).all().all():
            raise ValueError("Nonfinite labels")
        hashes[target] = sha256(source / f"task_records/{target}.csv")
        rgb_checkpoints[target] = sha256(paired.REFERENCE / f"runs/efficientnet_b0/{target}/model.pt")
        previous_path = source / f"runs/efficientnet_b0/{target}/video_predictions.csv"
        previous = records_at(previous_path).query("split == 'test'").sort_values("video_id").reset_index(drop=True)
        test = frame.loc[frame.split.eq("test")].sort_values("video_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(test[["hospital_id", "video_id"]], previous[["hospital_id", "video_id"]])
        np.testing.assert_allclose(test.raw_value, previous.y_true, rtol=1e-12, atol=1e-12)
        if not previous.frame_count.eq(20).all():
            raise RuntimeError("Four-channel comparator lost test frames")
        control_predictions[target] = sha256(previous_path)
        for split, part in frame.groupby("split"):
            counts.append({"target": target, "split": split, "videos": len(part), "patients": part.hospital_id.nunique(),
                           "lab_events": part.clinical_event_id.nunique(), "frame_pairs": len(part)*20})
    weight = Path(paired.WEIGHTS_DIR) / WEIGHT_FILES["efficientnet_b0"]
    contract = {"schema_version": 1, "architecture": "two independent ImageNet EN-B0 encoders; RGB1280 concat IR1280 -> head32 -> raw regression",
        "targets": list(config.TARGETS), "source": str(source), "lab_table_sha256": lab_hash,
        "records_sha256": hashes, "paired_protocol_sha256": sha256(source / "protocol.json"),
        "scalers_sha256": sha256(source / "target_scalers.json"), "selected_pairs_sha256": old["paired_table_sha256"],
        "rgb_index_sha256": sha256(rgb_path), "ir_index_sha256": sha256(ir_path),
        "index_source_manifest_sha256": sha256(config.paired.CACHE / "manifest.json"),
        "imagenet_weight_sha256": sha256(weight), "rgb_comparator_checkpoint_sha256": rgb_checkpoints,
        "four_channel_comparator_prediction_sha256": control_predictions,
        "initialization": "independent ImageNet parameters for both branches; no task-trained RGB initialization",
        "normalization": "both branches ImageNet mean/std; native gray IR repeated to 3 channels, no pseudo-color",
        "split_policy": "exact original per-analyte patients/splits; paired20 eligibility filter; no re-search",
        "scaling": "exact paired train-only median/IQR, no clipping", "job_seeds": old["job_seeds"],
        "matching_hours": 24, "frames": 20, "views": old["views"],
        "head_hidden": 32, "head_parameters": 82049, "total_parameters": 8097145,
        "training": {"protocol": "two_stage_full", "epochs": [40, 60], "patience": [10, 12], "lr": [2e-4, 1e-5],
            "minimum_lr": 1e-6, "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": "separate cosine per stage; no warmup",
            "stage1": "both encoders frozen and eval; head trainable", "stage2": "both full encoders and head trainable",
            "bn": "frozen in head stage; updated on microbatches in finetune (not bitwise full-batch BN)",
            "loss": "unweighted per-frame SmoothL1 beta=.5", "logical_batch": "12 distinct lab events x20 paired frames x1 view=240 pairs",
            "microbatch_pairs": config.MICROBATCH, "accumulation": "loss sum / actual logical-batch size; one update per logical batch, retain partials",
            "amp": "fp16", "gradient_clip": 1., "compile": True,
            "execution": "Inductor default mode, dynamic=True, CUDA Graphs disabled; eager evaluation; no silent fallback; compile_threads=1",
            "selection": "minimum video-level validation raw MAE across both stages"},
        "evaluation": "20 original synchronized frame pairs/video; mean standardized predictions, inverse train-only scaling",
        "comparison": "RGB main re-evaluated on exact paired RGB frames; main RGB training cohort is larger; 4-channel control shares paired cohort but not microbatch BN"}
    return rgb, scalers, pd.DataFrame(counts), contract


def initialize(queue):
    global RGB, IR, DEVICE
    gpu = int(queue.get())
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    RGB, IR = load_indexes(config.RGB_INDEX, config.IR_INDEX)
    print(f"[worker-ready] {DEVICE}; independent dual EN-B0; microbatch={config.MICROBATCH}", flush=True)


def rgb_comparator(target, records, path, device):
    source = config.paired.REFERENCE / f"runs/efficientnet_b0/{target}/model.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    scaler = RobustTargetScaler(**checkpoint["target_scaler"])
    test = records.loc[records.split.eq("test")].copy()
    test["robust_scaled_raw_value"] = scaler.transform(test.raw_value)
    dataset, loader = primary._loader(RGB, test, ("original",), "efficientnet_b0", False)
    model, _, _ = rgb_model("efficientnet_b0", config.paired.WEIGHTS_DIR)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device, memory_format=torch.channels_last)
    result = evaluate(model, loader, nn.SmoothL1Loss(beta=.5, reduction="none"), device)
    metrics, _, predictions = primary._video_metrics(result, dataset, "test", target, scaler)
    if not predictions.frame_count.eq(20).all() or len(predictions) != len(test):
        raise RuntimeError("RGB baseline lost test frames")
    predictions.to_csv(path / "main_rgb_same_frames_predictions.csv", index=False)
    pd.DataFrame([{**metrics, "reference_train_iqr": scaler.iqr}]).to_csv(path / "main_rgb_same_frames_metrics.csv", index=False)
    dataset.close()


def train_one(job, output=None, smoke=False):
    output = config.OUTPUT if output is None else output
    target = job["target"]
    records = records_at(output / f"task_records/{target}.csv")
    scaler = RobustTargetScaler(**json.loads((output / "target_scalers.json").read_text())[target])
    # Recompute from float64 raw values to avoid inherited CSV float rounding.
    records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
    _set_seed(job["seed"])
    path = output / f"regression/dual_enb0/runs/{target}"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(TrainingTee(sys.stdout, log)):
        with adapters(IR):
            primary.train_task("efficientnet_b0", target, RGB, records, scaler, config.paired.WEIGHTS_DIR, str(path),
                head_epochs=1 if smoke else 40, finetune_epochs=1 if smoke else 60,
                head_patience=10, finetune_patience=12, head_learning_rate=2e-4, finetune_learning_rate=1e-5,
                head_min_learning_rate=1e-6, finetune_min_learning_rate=1e-6, weight_decay=1e-4,
                train_batch_policy="distinct_lab_views", loss_level="frame", max_batches=1 if smoke else None)
        saved = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        saved.update(model_variant="independent_rgb_ir_dual_enb0", input_channels=6,
            channel_order=["R", "G", "B", "IR", "IR", "IR"], ir_initialization="independent ImageNet backbone",
            frame_policy="20 native224 FFV1 synchronous mutual-nearest RGB/IR pairs, <=33.333ms, source gaps>=2",
            execution="Inductor dynamic, no CUDA Graphs, no eager fallback; eager evaluation; microbatch60 accumulation to240 pairs",
            compile_enabled=True, seed=job["seed"],
            experiment_contract=job["contract"])
        torch.save(saved, path / "model.pt")
        del saved
        gc.collect()
        torch.cuda.empty_cache()
        rgb_comparator(target, records, path, DEVICE)
    if not smoke:
        (path / "job_complete.json").write_text(json.dumps({"contract": job["contract"]})+"\n")
    gc.collect()
    torch.cuda.empty_cache()
    return {"target": target, "seed": job["seed"], "status": "ok"}


def smoke(manifest, scalers):
    global RGB, IR, DEVICE
    RGB, IR = load_indexes(config.RGB_INDEX, config.IR_INDEX)
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    DEVICE = torch.device("cuda:0")
    target = "lactate_high"
    records = records_at(config.paired.OUTPUT / f"task_records/{target}.csv")
    train = records.loc[records.split.eq("train")].drop_duplicates("clinical_event_id").head(12)
    held = records.loc[records.split.ne("train")].groupby("split", group_keys=False).head(2)
    with tempfile.TemporaryDirectory(prefix="dual_enb0_smoke_") as name:
        output = Path(name)
        (output / "task_records").mkdir()
        pd.concat((train, held), ignore_index=True).to_csv(output / f"task_records/{target}.csv", index=False)
        (output / "target_scalers.json").write_text(json.dumps({target: scalers[target]}))
        train_one({"target": target, "seed": manifest["job_seeds"][target], "contract": "smoke"}, output, True)
        root = output / f"regression/dual_enb0/runs/{target}"
        history = pd.read_csv(root / "history.csv")
        if (len(history) != 2 or not history.train_model_inputs.eq(240).all() or not history.optimizer_steps.eq(1).all()
                or not history.execution_backend.eq("inductor:dynamic_no_cudagraphs").all()):
            raise RuntimeError("Smoke did not exercise one complete logical batch in both stages")
        pred = records_at(root / "video_predictions.csv")
        if len(pred) != 16 or not pred.frame_count.eq(20).all():
            raise RuntimeError("Smoke evaluation lost paired frames")
        frozen = torch.load(root / "stage_head_best.pt", map_location="cpu", weights_only=True)["model_state_dict"]
        tuned = torch.load(root / "stage_finetune_best.pt", map_location="cpu", weights_only=True)["model_state_dict"]
        from .models import build_model
        initial = build_model("efficientnet_b0", config.paired.WEIGHTS_DIR)[0].state_dict()
        for branch in ("rgb_encoder", "ir_encoder"):
            keys = [key for key in frozen if key.startswith(branch)]
            if any(not torch.equal(frozen[key], initial[key]) for key in keys):
                raise RuntimeError("Frozen head stage changed a backbone parameter or buffer")
            if not any(not torch.equal(tuned[key], frozen[key]) for key in keys if key.endswith("weight")):
                raise RuntimeError("Finetuning did not update both independent backbones")
        peak = history.peak_gpu_memory_gb.max()
    print(f"[smoke-ok] compiled forward/backward in both stages; 12labs/240pairs/checkpoints; "
          f"both backbones frozen then updated; peak={peak:.2f}GiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use1..8 GPU workers")
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, scalers, counts, manifest = prepare()
        print(counts.to_string(index=False), flush=True)
        if args.check_only:
            return
        protocol = config.OUTPUT / "protocol.json"
        if protocol.exists() and json.loads(protocol.read_text()) != manifest:
            raise RuntimeError("Different existing experiment protocol; refusing to overwrite")
        protocol.write_text(json.dumps(manifest, indent=2)+"\n")
        contract = sha256(protocol)
        if args.smoke and not (config.OUTPUT / "smoke_complete.json").exists():
            smoke(manifest, scalers)
            (config.OUTPUT / "smoke_complete.json").write_text(json.dumps({"contract": contract})+"\n")
        (config.OUTPUT / "task_records").mkdir(exist_ok=True)
        for target in config.TARGETS:
            shutil.copy2(config.paired.OUTPUT / f"task_records/{target}.csv", config.OUTPUT / f"task_records/{target}.csv")
        shutil.copy2(config.paired.OUTPUT / "target_scalers.json", config.OUTPUT / "target_scalers.json")
        counts.to_csv(config.OUTPUT / "dataset_counts.csv", index=False)
        jobs, statuses = [], []
        for target in config.TARGETS:
            marker = config.OUTPUT / f"regression/dual_enb0/runs/{target}/job_complete.json"
            if (marker.exists() and json.loads(marker.read_text()) == {"contract": contract}
                    and all((marker.parent / name).is_file() for name in ("model.pt", "stage_head_best.pt", "stage_finetune_best.pt",
                        "history.csv", "metrics.csv", "video_predictions.csv", "main_rgb_same_frames_predictions.csv", "main_rgb_same_frames_metrics.csv"))):
                statuses.append({"target": target, "status": "ok"})
            else:
                jobs.append({"target": target, "seed": manifest["job_seeds"][target], "contract": contract})
        workers = min(args.workers, torch.cuda.device_count(), len(jobs))
        print(f"[scheduler] independent dual EN-B0 tasks={len(jobs)} GPU workers={workers}", flush=True)
        (config.OUTPUT / "COMPLETE").unlink(missing_ok=True)
        if jobs:
            if not workers:
                raise RuntimeError("CUDA required")
            context = mp.get_context("spawn")
            with context.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize, initargs=(queue,)) as pool:
                    futures = {pool.submit(train_one, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            statuses.append(future.result())
                        except Exception:
                            statuses.append({"target": job["target"], "status": "failed", "error": traceback.format_exc()})
                            print(statuses[-1]["error"], flush=True)
                        pd.DataFrame(statuses).to_csv(config.OUTPUT / "run_index.csv", index=False)
        if len(statuses) != len(config.TARGETS) or any(row["status"] != "ok" for row in statuses):
            raise RuntimeError("Dual EN-B0 models failed; see run_index.csv")
        # Revalidate source videos, clinical files, scalers and baseline weights.
        if prepare()[3] != manifest:
            raise RuntimeError("Source protocol changed during training")
        from .plots import plot_results
        plot_results()
        (config.OUTPUT / "COMPLETE").write_text("Ten independent dual EN-B0 regressors, full histories and paired PNG comparisons complete\n")
        print("[queue-complete] independent RGB/IR EN-B0 two-stage regression", flush=True)


if __name__ == "__main__":
    main()
