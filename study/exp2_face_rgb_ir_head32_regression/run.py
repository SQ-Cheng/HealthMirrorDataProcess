"""Four-channel regression using the primary trainer and local scoped adapters."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import gc
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression import train as primary
from study.exp2_face_pretrained_head32_regression.run_all import _set_seed
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from preprocess.face_crop_comparison.run_comparison import sha256
from .config import HERE, ROOT, REFERENCE, PAIRED, OUTPUT, TARGETS, SCORE_DEFINITIONS, WEIGHTS_DIR
from .data import RGBIRDataset, prepare_images
from .models import build_model
from .index import build_indexes, load_indexes


RGB_INDEX = IR_INDEX = DEVICE = None


def safe_execution(model, architecture, target, stage, device):
    if device.type != "cuda" or not primary.TORCH_COMPILE_ENABLED:
        return model, "eager"
    compiled = torch.compile(model, dynamic=True, options={"triton.cudagraphs": False})
    print(f"[compile-enabled] {target}/{stage} four-channel dynamic batches; cudagraphs=False; evaluation=eager", flush=True)
    return compiled, "inductor:default_dynamic_no_cudagraphs"


@contextmanager
def adapters(ir_index, compile_enabled=True):
    names = ("AllFramesDataset", "_prepare_images", "build_pretrained_model", "_execution_model", "_evaluate", "TORCH_COMPILE_ENABLED")
    original = {name: getattr(primary, name) for name in names}
    primary.AllFramesDataset = lambda index, records, **kwargs: RGBIRDataset(index, records, ir_index, **kwargs)
    primary._prepare_images = prepare_images
    primary.build_pretrained_model = build_model
    primary._execution_model = safe_execution
    primary._evaluate = lambda model, *a, **kw: original["_evaluate"](getattr(model, "_orig_mod", model), *a, **kw)
    primary.TORCH_COMPILE_ENABLED = compile_enabled
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(primary, name, value)


def cleanup():
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare(output, targets, smoke=False):
    paired = json.loads((PAIRED / "summary.json").read_text())
    if not paired["complete_for_candidate_population"] or paired["quality_unknown_candidate_pairs"]:
        raise RuntimeError("Full IR quality/matching audit must be complete")
    table = pd.read_csv(PAIRED / "tables/video_pair_counts.csv")
    eligible = set(table.loc[table.pairs20_eligible, "video_id"])
    protocol = json.loads((REFERENCE / "main_protocol.json").read_text())
    extension = json.loads((REFERENCE / "direct_field_extension_protocol.json").read_text())
    expected = {**protocol["source_records_sha256"], **extension["record_hashes"][str(REFERENCE)]}
    if sha256(ROOT / "merged_lab_tests.csv") != protocol["lab_table_sha256"]:
        raise RuntimeError("Main reference is not based on the current lab table")
    seeds = pd.read_csv(REFERENCE / "run_index.csv").set_index("target").seed.to_dict()
    counts, scalers, hashes, videos, job_seeds = [], {}, {}, set(), {}
    (output / "task_records").mkdir(parents=True, exist_ok=True)
    for target in targets:
        path = REFERENCE / f"task_records/{target}.csv"
        hashes[target] = sha256(path)
        if hashes[target] != expected[target]:
            raise RuntimeError("Reference task records changed")
        baseline = pd.read_csv(path, dtype={"hospital_id": str}, float_precision="round_trip")
        records = baseline.loc[baseline.video_id.isin(eligible)].copy()
        if smoke:
            parts = []
            for split, n in (("train", 24), ("val", 4), ("test", 4)):
                pool = records.loc[records.split.eq(split)].drop_duplicates("hospital_id")
                parts.append(pool.sample(n=min(n, len(pool)), random_state=int(seeds[target])))
            records = pd.concat(parts, ignore_index=True)
        if records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any():
            raise ValueError("Duplicate videos or patient leakage")
        if set(records.split) != {"train", "val", "test"}:
            raise ValueError("All three original patient splits must survive")
        scaler = fit_robust_target_scaler(target, records, SCORE_DEFINITIONS[target]["unit"])
        records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
        records.to_csv(output / f"task_records/{target}.csv", index=False)
        scalers[target] = scaler.to_dict()
        videos.update(records.video_id)
        job_seeds[target] = int(seeds[target])
        for split, group in records.groupby("split"):
            counts.append({"target": target, "split": split, "videos": len(group),
                "patients": group.hospital_id.nunique(), "clinical_events": group.clinical_event_id.nunique(),
                "main_videos": int(baseline.split.eq(split).sum())})
        print(f"[prepare] {target}: paired videos={len(records)} patients={records.hospital_id.nunique()}", flush=True)
    pd.DataFrame(counts).to_csv(output / "dataset_counts.csv", index=False)
    (output / "target_scalers.json").write_text(json.dumps(scalers, indent=2)+"\n")
    rgb_index, ir_index = build_indexes(videos)
    manifest = {"experiment": HERE.name, "targets": targets, "reference": str(REFERENCE),
        "reference_record_hashes": hashes, "lab_table_sha256": protocol["lab_table_sha256"], "job_seeds": job_seeds,
        "split_policy": "keep exact per-analyte main assignments; filter paired20 eligibility; no new split search",
        "label_policy": "exact main nearest24h labels; no rematching", "input_channels": ["R", "G", "B", "IR"],
        "stem": "pretrained Conv2d(3,32,3,stride2) -> Conv2d(4,32,3,stride2); RGB filters copied, IR slice zero",
        "parameters": 4048925, "additional_stem_parameters": 288, "head_hidden_features": 32,
        "normalization": {"rgb": "unchanged ImageNet mean/std", "ir": "fixed mean=.5 std=.5; no held-out statistics"},
        "views": ["original", "hflip", "center_crop", "brightness", "contrast"],
        "augmentation": "same view and parameters on all four channels; per-channel contrast means",
        "spatial_registration": "unchanged independent MediaPipe crops; no extra cross-modal registration",
        "frames": 20, "paired_table_sha256": sha256(PAIRED / "tables/selected20_pairs.csv"),
        "rgb_index": str(rgb_index), "ir_index": str(ir_index),
        "training": {"head_lr": 2e-4, "finetune_lr": 1e-5, "epochs": [40, 60], "patience": [10, 12],
            "min_lr": 1e-6, "weight_decay": 1e-4, "optimizer": "AdamW", "scheduler": "cosine, no warmup",
            "loss": "unweighted per-frame SmoothL1(beta=.5)", "batch": "12 distinct lab events x20 frames x1 view=240",
            "stage1": "head only; frozen zero IR stem means RGB-only prediction", "stage2": "all parameters unfrozen, including IR stem slice",
            "target_scaling": "same train-only median/IQR algorithm refit on the surviving train cohort",
            "execution": "Inductor default dynamic, no CUDA Graphs; eager evaluation; caches cleared per job"},
        "comparison": "common test videos and original raw labels; existing RGB model also evaluated on exact joint-selected RGB frames; training cohort difference disclosed",
        "smoke_test": smoke}
    if smoke:
        manifest["training"]["epochs"] = [1, 1]
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def initialize(queue, rgb_path, ir_path):
    global RGB_INDEX, IR_INDEX, DEVICE
    DEVICE = int(queue.get())
    torch.cuda.set_device(DEVICE)
    torch.set_num_threads(1)
    RGB_INDEX, IR_INDEX = load_indexes(rgb_path, ir_path)
    print(f"[worker-ready] cuda:{DEVICE} paired_frames={len(RGB_INDEX.starts)}", flush=True)


def baseline_same_frames(target, records, path):
    checkpoint = torch.load(REFERENCE / f"runs/efficientnet_b0/{target}/model.pt", map_location="cpu", weights_only=True)
    scaler = RobustTargetScaler(**checkpoint["target_scaler"])
    test = records.loc[records.split.eq("test")].copy()
    test["robust_scaled_raw_value"] = scaler.transform(test.raw_value)
    dataset, loader = primary._loader(RGB_INDEX, test, ("original",), "efficientnet_b0", False)
    model, _, _ = primary.build_pretrained_model("efficientnet_b0", WEIGHTS_DIR)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(f"cuda:{DEVICE}")
    model = model.to(device, memory_format=torch.channels_last)
    evaluation = primary._evaluate(model, loader, nn.SmoothL1Loss(beta=.5, reduction="none"), device)
    metrics, _, predictions = primary._video_metrics(evaluation, dataset, "test", target, scaler)
    predictions.to_csv(path / "main_rgb_same_frames_predictions.csv", index=False)
    pd.DataFrame([{**metrics, "target": target, "model": "RGB main, joint frames", "reference_train_iqr": scaler.iqr}]).to_csv(path / "main_rgb_same_frames_metrics.csv", index=False)
    dataset.close()
    del model, loader, dataset


def job_task(job):
    cleanup()
    try:
        output, target = Path(job["output"]), job["target"]
        records = pd.read_csv(output / f"task_records/{target}.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        scaler = RobustTargetScaler(**json.loads((output / "target_scalers.json").read_text())[target])
        records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
        _set_seed(job["seed"])
        path = output / f"runs/efficientnet_b0/{target}"
        with adapters(IR_INDEX, compile_enabled=not job["smoke"]):
            primary.train_task("efficientnet_b0", target, RGB_INDEX, records, scaler, WEIGHTS_DIR, str(path),
                head_epochs=1 if job["smoke"] else 40, finetune_epochs=1 if job["smoke"] else 60,
                train_batch_policy="distinct_lab_views", loss_level="frame", max_batches=1 if job["smoke"] else None)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        checkpoint.update(input_channels=4, channel_order=["R", "G", "B", "IR"], ir_initialization="zero",
                          ir_normalization={"mean": .5, "std": .5}, frame_policy="20 synchronous native224 RGB+IR frames, mutual nearest <=33.333ms",
                          execution="default dynamic Inductor without CUDA Graphs; eager evaluation")
        torch.save(checkpoint, path / "model.pt")
        verifier, _, _ = build_model("efficientnet_b0", WEIGHTS_DIR)
        verifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del verifier, checkpoint
        cleanup()
        baseline_same_frames(target, records, path)
        return target
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-prepared", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    targets = tuple(args.targets.split(","))
    if not set(targets).issubset(TARGETS) or len(set(targets)) != len(targets):
        raise ValueError("Unknown or repeated targets")
    output = args.output_dir.resolve()
    if args.smoke_test and output == OUTPUT:
        output = HERE / "outputs_smoke"
    if args.train_prepared:
        manifest = json.loads((output / "protocol.json").read_text())
        if manifest["targets"] != list(targets) or manifest["smoke_test"] != args.smoke_test or (output / "run_index.csv").exists():
            raise ValueError("Prepared protocol differs or training has started")
        if manifest["lab_table_sha256"] != sha256(ROOT / "merged_lab_tests.csv"):
            raise RuntimeError("Lab table changed")
        for target, value in manifest["reference_record_hashes"].items():
            if value != sha256(REFERENCE / f"task_records/{target}.csv"):
                raise RuntimeError("Reference split/labels changed")
        build_indexes([v for target in targets for v in pd.read_csv(output / f"task_records/{target}.csv").video_id])
    else:
        if (output / "protocol.json").exists():
            raise FileExistsError("Use a fresh output directory or --train-prepared; never overwrite main results")
        manifest = prepare(output, targets, args.smoke_test)
    if args.prepare_only:
        return
    devices = [int(value) for value in args.devices.split(",")]
    if not torch.cuda.is_available() or len(set(devices)) != len(devices) or any(d < 0 or d >= torch.cuda.device_count() for d in devices):
        raise ValueError("Valid distinct CUDA devices required")
    context = mp.get_context("spawn")
    queue = context.Queue()
    for device in devices:
        queue.put(device)
    statuses = []
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context, initializer=initialize,
                             initargs=(queue, manifest["rgb_index"], manifest["ir_index"])) as pool:
        futures = {pool.submit(job_task, {"target": t, "output": str(output), "seed": manifest["job_seeds"][t], "smoke": args.smoke_test}): t for t in targets}
        for future in as_completed(futures):
            target = futures[future]
            try:
                future.result(); statuses.append({"target": target, "status": "ok"})
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True); statuses.append({"target": target, "status": "failed", "error": error})
            pd.DataFrame(statuses).to_csv(output / "run_index.csv", index=False)
            print(f"[progress] {len(statuses)}/{len(targets)} {target} {statuses[-1]['status']}", flush=True)
    if any(row["status"] != "ok" for row in statuses):
        raise RuntimeError("Some RGB+IR models failed; no complete result is declared")
    from .plots import plot_results
    plot_results(output, targets)
    (output / "COMPLETED.json").write_text(json.dumps({"targets": targets, "completed_utc": pd.Timestamp.now(tz="UTC").isoformat()})+"\n")


if __name__ == "__main__":
    main()
