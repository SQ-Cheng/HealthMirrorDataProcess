"""Verify the shared paired data, cache frozen features, then fit sixty heads."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from study.common.run_video_loss_12h import Tee, sha256
from study.exp2_face_architecture_ablation.train import train_task
from study.exp2_face_dinov3_frozen.data import no_feature_fitting
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from study.exp2_face_rgb_ir_head32_regression.index import build_indexes, load_indexes
from . import config
from .data import loader
from .features import cache_contract, ensure_cache
from .models import build_model


INDEX = DEVICE = None


def read_records(path):
    return pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")


def validate_records(records, original, scaler, index):
    if (records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any()
            or set(records.split) != {"train", "val", "test"} or records.clinical_event_id.isna().any()
            or not records.match_delta_h.between(0, 24+1e-9).all()):
        raise ValueError("Invalid paired patient split/labels/matching window")
    inherited = original.set_index("video_id").loc[records.video_id].reset_index()
    for column in ("hospital_id", "split", "clinical_event_id"):
        np.testing.assert_array_equal(records[column], inherited[column])
    np.testing.assert_allclose(records.raw_value, inherited.raw_value, rtol=1e-12, atol=1e-12)
    fitted = fit_robust_target_scaler(scaler["target"], records, scaler["unit"])
    if fitted.to_dict() != scaler:
        raise ValueError("Scaler was not fitted on exactly the paired training videos")
    transformed = RobustTargetScaler(**scaler).transform(records.raw_value)
    np.testing.assert_allclose(records.robust_scaled_raw_value, transformed, rtol=1e-6, atol=1e-6)
    if not np.isfinite(records[["raw_value", "robust_scaled_raw_value"]]).all().all():
        raise ValueError("Nonfinite paired regression labels")
    training = records.loc[records.split.eq("train")]
    if training.groupby("clinical_event_id").raw_value.nunique().gt(1).any():
        raise ValueError("One clinical event has inconsistent values")
    for video in records.video_id:
        if np.diff(index.video_ptr)[index.video_lookup[video]] != 20:
            raise ValueError("Exactly twenty synchronized frame pairs required")


def prepare():
    config.reference.configure_variant(32, "dinov3_vitb16")
    _, _, main_hashes, _, lab_hash = config.reference.preflight()
    source = config.paired.OUTPUT
    protocol = json.loads((source / "protocol.json").read_text())
    if protocol["smoke_test"] or protocol["lab_table_sha256"] != lab_hash or tuple(protocol["targets"]) != tuple(config.TARGETS):
        raise RuntimeError("Paired reference is not the current full ten-task cohort")
    if protocol["paired_table_sha256"] != sha256(config.paired.PAIRED / "tables/selected20_pairs.csv"):
        raise RuntimeError("Selected synchronous frame pairs changed")
    summary = json.loads((config.paired.PAIRED / "summary.json").read_text())
    if not summary["complete_for_candidate_population"] or summary["quality_unknown_candidate_pairs"]:
        raise RuntimeError("IR preprocessing/pair audit is incomplete")
    scalers = json.loads((source / "target_scalers.json").read_text())
    records = {target: read_records(source / f"task_records/{target}.csv") for target in config.TARGETS}
    rgb_path, ir_path = build_indexes(video for frame in records.values() for video in frame.video_id)
    if rgb_path != config.RGB_INDEX or ir_path != config.IR_INDEX:
        raise RuntimeError("Unexpected paired index paths")
    index, _ = load_indexes(rgb_path, ir_path)
    counts, hashes, seeds = [], {}, {}
    for target, frame in records.items():
        if protocol["reference_record_hashes"][target] != main_hashes[target]:
            raise RuntimeError("Paired cohort labels/splits refer to a different main experiment")
        validate_records(frame, read_records(config.paired.REFERENCE / f"task_records/{target}.csv"), scalers[target], index)
        hashes[target] = sha256(source / f"task_records/{target}.csv")
        seeds[target] = config.reference.reference.job_seed("regression", target)
        if seeds[target] != protocol["job_seeds"][target]:
            raise RuntimeError("Seed differs from paired reference")
        for split, part in frame.groupby("split"):
            counts.append({"target": target, "split": split, "videos": len(part), "patients": part.hospital_id.nunique(),
                           "clinical_events": part.clinical_event_id.nunique(), "paired_frames": len(part)*20})
    contract = {"schema_version": 1, "targets": list(config.TARGETS), "variants": list(config.VARIANTS),
        "reference": str(source), "records_sha256": hashes, "lab_table_sha256": lab_hash,
        "paired_protocol_sha256": sha256(source / "protocol.json"), "scalers_sha256": sha256(source / "target_scalers.json"),
        "paired_selection_sha256": protocol["paired_table_sha256"], "pair_policy": "mutual nearest, unique IR, <=33.333ms, source index gaps>=2 in both modalities",
        "feature_contract": cache_contract(), "job_seeds": seeds,
        "parameters": {variant: sum(p.numel() for p in build_model(variant).parameters()) for variant in config.VARIANTS},
        "head_hidden": 32, "projection_dimension": 64, "gate": "sigmoid RGB weight, zero initialized (.5); IR weight=1-RGB; 64-vector or single broadcast scalar",
        "dino_farl_policy": "concat DINO768/FaRL512 within each modality, then fuse RGB1280/IR1280",
        "split_policy": "exact main per-analyte patient assignments, filter paired20; no re-search; all six structures identical",
        "target_scaling": "paired train-only raw median/IQR, identical six structures; no target clipping",
        "matching_hours": 24, "frames_per_video": 20, "views": list(config.VIEWS),
        "batch": "12 distinct lab events x20 synchronized frame pairs x1 shared view=240 frame pairs; partial batch retained",
        "loss_level": "frame", "loss": "unweighted SmoothL1 beta=.5 on robust-scaled raw values",
        "evaluation": "20 original frame pairs/video; mean scaled prediction then inverse scaler; video-level metrics",
        "training": {"stages": 1, "epochs": 80, "patience": 12, "optimizer": "AdamW", "lr": 2e-4,
                     "min_lr": 1e-6, "weight_decay": .001, "scheduler": "cosine, no warmup", "dropout": .25,
                     "gradient_clip": 1., "amp": "fp16", "compile": False,
                     "selection": "lowest video-level validation raw MAE; never select on test"},
        "frozen_policy": "independent RGB and IR modules per pretrained family; all encoders eval/no_grad, only fusion/head optimized",
        "cache_policy": "label-independent frozen representations shared across tasks; no feature statistics fitted on held-out data"}
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print(f"[paired-cohort] videos={len(index.video_ids)} patients={pd.concat(list(records.values())).hospital_id.nunique()} heads=60", flush=True)
    return index, scalers, pd.DataFrame(counts), contract


def training_config(output=None, epochs=80):
    return SimpleNamespace(HERE=config.HERE, OUTPUT_DIR=config.OUTPUT if output is None else output,
        reference=config.reference.reference, LOSS_LEVEL="frame",
        LEARNING_RATES={variant: 2e-4 for variant in config.VARIANTS},
        MIN_LEARNING_RATES={variant: 1e-6 for variant in config.VARIANTS}, MAX_EPOCHS=epochs, PATIENCE=12, WEIGHT_DECAY=.001)


def init_worker(queue):
    global INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX, _ = load_indexes(config.RGB_INDEX, config.IR_INDEX)
    print(f"[head-worker] cuda:{gpu} ready", flush=True)


def worker(job):
    root = config.OUTPUT / f"regression/{job['architecture']}/runs/{job['target']}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_task(job, INDEX, DEVICE, experiment_config=training_config(), model_factory=build_model,
                   loader_factory=loader, feature_scaler=no_feature_fitting)
    path = root / "model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved.update(experiment_contract=job["contract"], frozen_encoders=True, head_hidden=32,
                 feature_branches=list(config.branches_for(job["architecture"])), matching_hours=24,
                 feature_manifest_sha256=job["cache_sha256"], source_records_sha256=job["source_sha256"])
    torch.save(saved, path)
    (root / "job_complete.json").write_text(json.dumps({"contract": job["contract"]})+"\n")
    torch.cuda.empty_cache()
    return {"architecture": job["architecture"], "target": job["target"], "status": "ok", "seed": saved["seed"]}


def smoke(index, scalers):
    target = "lactate_high"
    full = read_records(config.paired.OUTPUT / f"task_records/{target}.csv")
    train = full.loc[full.split.eq("train")].drop_duplicates("clinical_event_id").head(12)
    held = full.loc[full.split.ne("train")].groupby("split", group_keys=False).head(2)
    subset = pd.concat((train, held), ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="paired_frozen_smoke_") as name:
        output = Path(name)
        (output / "source_records").mkdir()
        subset.to_csv(output / f"source_records/{target}.csv", index=False)
        for variant in config.VARIANTS:
            train_task({"architecture": variant, "target": target, "family": "regression", "scaler": scalers[target]},
                index, torch.device("cuda:0"), experiment_config=training_config(output, 1),
                model_factory=build_model, loader_factory=loader, feature_scaler=no_feature_fitting)
            predictions = read_records(output / f"regression/{variant}/runs/{target}/video_predictions.csv")
            assert predictions.frame_count.eq(20).all() and len(predictions) == len(subset)
    print("[smoke-ok] all six heads: real features, full 12-lab batch, all5 views, backward, checkpoint, video evaluation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a bounded real-data smoke before the full queue")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1..8 GPU workers")
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index, scalers, counts, manifest = prepare()
        if args.check_only:
            return
        path = config.OUTPUT / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing experiment contract differs; do not overwrite results")
        path.write_text(json.dumps(manifest, indent=2)+"\n")
        contract = sha256(path)
        cache_sha256 = ensure_cache(args.workers)
        if args.smoke and not (config.OUTPUT / "smoke_complete.json").exists():
            smoke(index, scalers)
            (config.OUTPUT / "smoke_complete.json").write_text(json.dumps({"contract": contract, "features": cache_sha256})+"\n")
        (config.OUTPUT / "source_records").mkdir(exist_ok=True)
        for target in config.TARGETS:
            shutil.copy2(config.paired.OUTPUT / f"task_records/{target}.csv", config.OUTPUT / f"source_records/{target}.csv")
        shutil.copy2(config.paired.OUTPUT / "target_scalers.json", config.OUTPUT / "target_scalers.json")
        counts.to_csv(config.OUTPUT / "cohort_counts.csv", index=False)
        jobs, rows = [], []
        for target in config.TARGETS:
            for variant in config.VARIANTS:
                root = config.OUTPUT / f"regression/{variant}/runs/{target}"
                marker = root / "job_complete.json"
                if marker.exists() and json.loads(marker.read_text()) == {"contract": contract} and all((root / file).is_file() for file in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv")):
                    rows.append({"architecture": variant, "target": target, "status": "ok"})
                else:
                    jobs.append({"architecture": variant, "target": target, "family": "regression", "scaler": scalers[target],
                                 "contract": contract, "cache_sha256": cache_sha256, "source_sha256": manifest["records_sha256"][target]})
        workers = min(args.workers, torch.cuda.device_count(), len(jobs))
        print(f"[scheduler] remaining={len(jobs)} workers={workers}; dynamic next-job allocation across all visible GPUs", flush=True)
        (config.OUTPUT / "COMPLETE").unlink(missing_ok=True)
        if jobs:
            if not workers:
                raise RuntimeError("CUDA required")
            context = mp.get_context("spawn")
            with context.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(queue,)) as pool:
                    futures = {pool.submit(worker, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            rows.append(future.result())
                        except Exception:
                            rows.append({"architecture": job["architecture"], "target": job["target"], "status": "failed", "error": traceback.format_exc()})
                            print(rows[-1]["error"], flush=True)
                        pd.DataFrame(rows).to_csv(config.OUTPUT / "run_index.csv", index=False)
        if len(rows) != 60 or any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Incomplete heads; see run_index.csv and individual train.log")
        for target in config.TARGETS:
            if sha256(config.paired.OUTPUT / f"task_records/{target}.csv") != manifest["records_sha256"][target]:
                raise RuntimeError("Source records changed during training")
        if (sha256(config.paired.ROOT / "merged_lab_tests.csv") != manifest["lab_table_sha256"]
                or cache_contract() != manifest["feature_contract"]):
            raise RuntimeError("Source lab table or encoder/index contract changed")
        for variant in config.VARIANTS:
            root = config.OUTPUT / f"regression/{variant}"
            for field in ("metrics", "history"):
                pd.concat([pd.read_csv(root / f"runs/{target}/{field}.csv") for target in config.TARGETS], ignore_index=True).to_csv(root / f"{field}_all.csv", index=False)
        from .plots import plot_results
        plot_results()
        from .compare_rgb import main as compare_rgb
        compare_rgb()
        (config.OUTPUT / "COMPLETE").write_text("60 frozen RGB/IR regressors, all histories/checkpoints and paired PNG comparisons complete\n")
        print("[queue-complete] six architectures x ten analytes and all comparisons", flush=True)


if __name__ == "__main__":
    main()
