"""Frozen DINO heads on the exact ten-task 24h frame-loss regression cohort."""

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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from study.common import run_face_main_24h as main_reference
from study.common import run_video_loss_12h as reference
from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.exp2_face_architecture_ablation.train import train_task
from study.exp2_face_pretrained_head32_regression import config as baseline_config
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.scaling import fit_robust_target_scaler

from . import config
from .backbone import Head, ensure_source, weight_sha256
from .data import DinoFeatureDataset, no_feature_fitting
from .features import ensure_cache


BASELINE = Path(baseline_config.OUTPUT_DIRS["20frame"])
OUTPUT = config.HERE / "outputs/main24h_frame_loss"
RESULTS = OUTPUT / "regression" / config.ARCHITECTURE
CACHE = config.HERE / "cache/main24h_frame_loss"
INDEX_PATH = main_reference.INDEX_PATH
TARGETS = baseline_config.ALL_REGRESSION_TARGETS
INDEX = DEVICE = None
HEAD_HIDDEN = 32
HEAD_PARAMETERS = 12417


def configure_variant(head_hidden):
    global HEAD_HIDDEN, HEAD_PARAMETERS, OUTPUT, RESULTS
    if head_hidden not in (32, 64):
        raise ValueError("Supported DINO head widths are 32 and 64")
    HEAD_HIDDEN = head_hidden
    HEAD_PARAMETERS = 388 * head_hidden + 1
    suffix = "" if head_hidden == 32 else "_head64"
    OUTPUT = config.HERE / f"outputs/main24h_frame_loss{suffix}"
    RESULTS = OUTPUT / "regression" / config.ARCHITECTURE


def build_main_head(architecture):
    return Head(HEAD_HIDDEN)


def read_records(path):
    return pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})


def preflight():
    if not (BASELINE / "COMPLETE").is_file():
        raise RuntimeError("Main regression results are not complete")
    protocol = json.loads((BASELINE / "main_protocol.json").read_text())
    extension = json.loads((BASELINE / "direct_field_extension_protocol.json").read_text())
    lab_hash = reference.sha256(BASELINE.parents[3] / "merged_lab_tests.csv")
    if protocol["lab_table_sha256"] != lab_hash or extension["lab_table_sha256"] != lab_hash:
        raise RuntimeError("Main regression does not use the current lab table")
    expected = {"matching_hours": 24, "loss_level": "frame", "train_batch_policy": "distinct_lab_views",
                "frame_batch_size": 240, "batch_unique_measurements": 12, "frames_per_video": 20}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Unexpected main regression protocol")
    if reference.sha256(INDEX_PATH) != protocol["frame_index_sha256"]:
        raise RuntimeError("Main frame index changed")
    index = FrameOffsetIndex.load(INDEX_PATH)
    if set(index.video_formats) != {"ffv1"} or not _index_is_reusable(INDEX_PATH.parent, index.video_ids, "20frame"):
        raise RuntimeError("Native224 main frame index is stale")
    # HCT/eGFR used a larger index; compare actual packets, not row offsets.
    extension_path = Path(extension["frame_index"])
    if reference.sha256(extension_path) != extension["frame_index_sha256"]:
        raise RuntimeError("Direct-field frame index changed")
    extra_index = FrameOffsetIndex.load(extension_path)
    hashes = {**protocol["source_records_sha256"], **extension["record_hashes"][str(BASELINE)]}
    scalers = json.loads((BASELINE / "target_scalers.json").read_text())["targets"]
    audits = []
    metrics = pd.read_csv(BASELINE / "metrics_all.csv")
    for target in TARGETS:
        path = BASELINE / f"task_records/{target}.csv"
        if reference.sha256(path) != hashes[target]:
            raise RuntimeError(f"Main task records changed: {target}")
        records = read_records(path)
        if (records.video_id.duplicated().any() or set(records.split) != {"train", "val", "test"}
                or records.groupby("hospital_id").split.nunique().gt(1).any()
                or not records.match_delta_h.between(0, 24 + 1e-9).all()
                or records.clinical_event_id.isna().any()
                or not np.isfinite(records[["raw_value", "robust_scaled_raw_value"]]).all().all()):
            raise RuntimeError(f"Invalid main records: {target}")
        fitted = fit_robust_target_scaler(target, records, baseline_config.SCORE_DEFINITIONS[target]["unit"])
        saved = scalers[target]
        if (fitted.train_video_ids_sha256 != saved["train_video_ids_sha256"]
                or fitted.train_videos != saved["train_videos"]):
            raise RuntimeError(f"Scaler training population differs: {target}")
        np.testing.assert_allclose([fitted.median, fitted.q1, fitted.q3, fitted.iqr],
                                   [saved[key] for key in ("median", "q1", "q3", "iqr")], rtol=1e-12, atol=1e-12)
        transformed = reference.RobustTargetScaler(**saved).transform(records.raw_value).astype(np.float32)
        np.testing.assert_array_equal(transformed, records.robust_scaled_raw_value.to_numpy(np.float32))
        for video in records.video_id:
            start, end = index.frame_range(video)
            if end - start != 20 or np.diff(index.source_indices[start:end]).min() < 2:
                raise RuntimeError(f"Wrong selected-frame coverage: {video}")
            if target in baseline_config.ADDITIONAL_REGRESSION_TARGETS:
                a, b = index.video_lookup[video], extra_index.video_lookup[video]
                if (index.video_paths[a] != extra_index.video_paths[b]
                        or index.codec_extradata[a] != extra_index.codec_extradata[b]):
                    raise RuntimeError("Direct-field video source differs")
                for name in ("starts", "ends", "source_indices"):
                    np.testing.assert_array_equal(getattr(index, name)[start:end],
                                                  getattr(extra_index, name)[slice(*extra_index.frame_range(video))])
        run = BASELINE / f"runs/efficientnet_b0/{target}"
        prediction = read_records(run / "video_predictions.csv").sort_values("video_id").reset_index(drop=True)
        actual = records.sort_values("video_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(prediction[["hospital_id", "video_id", "split"]],
                                      actual[["hospital_id", "video_id", "split"]])
        np.testing.assert_array_equal(prediction.y_true, actual.raw_value)
        if not prediction.frame_count.eq(20).all() or set(metrics.loc[metrics.target.eq(target), "split"]) != {"train", "val", "test"}:
            raise RuntimeError(f"Incomplete baseline evaluation: {target}")
        for split, group in records.groupby("split"):
            audits.append({"target": target, "split": split, "videos": len(group),
                           "patients": group.hospital_id.nunique(), "lab_events": group.clinical_event_id.nunique()})
        training = records.loc[records.split.eq("train")].reset_index(drop=True)
        if training.groupby("clinical_event_id").raw_value.nunique().gt(1).any():
            raise RuntimeError("Inconsistent labels for one clinical event")
        dataset = SimpleNamespace(expand_all_views=False, video_records=training,
                                  frame_video_rows=np.repeat(np.arange(len(training)), 20), views=config.VIEWS)
        torch.manual_seed(reference.job_seed("regression", target))
        batches = list(DistinctLabViewBatchSampler(dataset, 240))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(training) * 100))
        for batch in batches:
            groups = np.asarray(batch).reshape(-1, 20)
            if (training.iloc[groups[:, 0] // 100].clinical_event_id.nunique() != len(groups)
                    or any(len(set(group // 100)) != 1 or len(set(group % 5)) != 1 for group in groups)):
                raise RuntimeError("Mixed views or repeated lab event in batch")
        if any(len(batch) != 240 for batch in batches[:-1]):
            raise RuntimeError("Unexpected short training batch")
    print("[preflight-ok] exact ten-task main labels/splits/scalers/frames; 24h; frame loss; 12 distinct labs", flush=True)
    return index, scalers, hashes, pd.DataFrame(audits), lab_hash


def training_config(output=None, max_epochs=None):
    output = OUTPUT if output is None else output
    return SimpleNamespace(HERE=config.HERE, OUTPUT_DIR=output, reference=reference,
                           LOSS_LEVEL="frame", LEARNING_RATES=config.LEARNING_RATES,
                           MIN_LEARNING_RATES=config.MIN_LEARNING_RATES,
                           MAX_EPOCHS=config.MAX_EPOCHS if max_epochs is None else max_epochs,
                           PATIENCE=config.PATIENCE, WEIGHT_DECAY=config.WEIGHT_DECAY)


def loader(index, records, architecture, family, train):
    dataset = DinoFeatureDataset(index, records, config.VIEWS if train else ("original",), family, CACHE)
    sampler = DistinctLabViewBatchSampler(dataset, 240) if train else VideoViewBatchSampler(dataset, 500, False)
    return dataset, DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)


def init_worker(queue, head_hidden=32):
    global INDEX, DEVICE
    configure_variant(head_hidden)
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX = FrameOffsetIndex.load(INDEX_PATH)


def worker(job):
    run = RESULTS / "runs" / job["target"]
    run.mkdir(parents=True, exist_ok=True)
    with (run / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(reference.Tee(sys.stdout, log)):
        train_task(job, INDEX, DEVICE, experiment_config=training_config(), model_factory=build_main_head,
                   loader_factory=loader, feature_scaler=no_feature_fitting)
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if saved["loss_unit"] != "frame" or saved["parameters"] != HEAD_PARAMETERS:
        raise RuntimeError("Incorrect DINO checkpoint protocol")
    saved.update(backbone_frozen=True, backbone_weight_sha256=job["weight_sha256"],
                 backbone_source_revision=config.REPO_REVISION, feature="normalized CLS token",
                 matching_hours=24, head_hidden=HEAD_HIDDEN, source_records_sha256=job["records_sha256"],
                 frame_index_sha256=reference.sha256(INDEX_PATH), train_batch_policy="distinct_lab_views")
    torch.save(saved, run / "model.pt")
    (run / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}) + "\n")
    gc.collect()
    torch.cuda.empty_cache()
    return {"architecture": config.ARCHITECTURE, "family": "regression", "target": job["target"],
            "status": "ok", "seed": saved["seed"]}


def smoke(index, scalers):
    target = "lactate_high"
    records = read_records(BASELINE / f"task_records/{target}.csv")
    training = records.loc[records.split.eq("train")].drop_duplicates("clinical_event_id").groupby("binary_label", group_keys=False).head(6)
    if len(training) != 12:
        raise RuntimeError("Smoke test needs twelve distinct lab events")
    held = records.loc[records.split.ne("train")].groupby("split", group_keys=False).head(2)
    subset = pd.concat([training, held], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="dino_main24h_smoke_") as name:
        root = Path(name)
        (root / "source_records").mkdir()
        subset.to_csv(root / f"source_records/{target}.csv", index=False)
        train_task({"architecture": config.ARCHITECTURE, "family": "regression", "target": target,
                    "scaler": scalers[target]}, index, torch.device("cuda:0"),
                   experiment_config=training_config(root, 1), model_factory=build_main_head,
                   loader_factory=loader, feature_scaler=no_feature_fitting)
        run = root / "regression" / config.ARCHITECTURE / "runs" / target
        saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        history = pd.read_csv(run / "history.csv")
        assert saved["loss_unit"] == "frame" and saved["parameters"] == HEAD_PARAMETERS
        assert int(history.train_model_inputs.iloc[0]) == 1200
        predictions = read_records(run / "video_predictions.csv")
        assert predictions.frame_count.eq(20).all()
    print("[smoke-ok] real cached features; 240 frame losses; optimizer/checkpoint/evaluation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--head-hidden", type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    configure_variant(args.head_hidden)
    ensure_source()
    index, scalers, hashes, counts, lab_hash = preflight()
    digest = weight_sha256()
    if args.check_only:
        print(counts.to_string(index=False))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {"targets": list(TARGETS), "baseline": str(BASELINE), "lab_table_sha256": lab_hash,
                    "matching_hours": 24, "source_records_sha256": hashes,
                    "frame_index_sha256": reference.sha256(INDEX_PATH),
                    "scalers_sha256": reference.sha256(BASELINE / "target_scalers.json"),
                    "baseline_predictions_sha256": {target: reference.sha256(BASELINE / f"runs/efficientnet_b0/{target}/video_predictions.csv") for target in TARGETS},
                    "architecture": config.ARCHITECTURE, "backbone_weight_sha256": digest,
                    "backbone_source_revision": config.REPO_REVISION, "backbone_frozen": True,
                    "head_parameters": HEAD_PARAMETERS, "head_hidden": HEAD_HIDDEN, "loss_level": "frame",
                    "loss": "unweighted per-frame SmoothL1(beta=0.5); train-only median/IQR raw target scaling",
                    "frame_batch_size": 240, "batch_distinct_measurements": 12,
                    "frames_per_video": 20, "training_views": list(config.VIEWS),
                    "views_per_video_per_batch": 1, "evaluation": "mean of 20 original-view frame predictions",
                    "job_seeds": {target: reference.job_seed("regression", target) for target in TARGETS},
                    "learning_rate": 2e-4, "minimum_lr": 1e-6, "max_epochs": 80,
                    "patience": 12, "weight_decay": 1e-3, "dropout": .25,
                    "optimizer": "AdamW", "scheduler": "cosine; no warmup", "stages": 1,
                    "comparison_note": "Identical cohorts/splits/frames/views/scalers/batch/loss/evaluation and seeds. Frozen DINO head-only optimization versus two-stage fine-tuned EfficientNet; not a backbone-only ablation."}
        head32_root = None
        if HEAD_HIDDEN == 64:
            head32_output = config.HERE / "outputs/main24h_frame_loss"
            if not (head32_output / "COMPLETE").is_file():
                raise RuntimeError("Head32 run is incomplete")
            original = json.loads((head32_output / "experiment_manifest.json").read_text())
            if any(manifest.get(key) != value for key, value in original.items()
                   if key not in ("head_hidden", "head_parameters")):
                raise RuntimeError("Head64 must differ from head32 only in hidden width")
            head32_root = head32_output / "regression" / config.ARCHITECTURE
            manifest["head32_baseline_predictions_sha256"] = {
                target: reference.sha256(head32_root / f"runs/{target}/video_predictions.csv") for target in TARGETS}
        path = OUTPUT / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing DINO main experiment contract differs")
        if args.plot_only:
            if not path.exists():
                raise RuntimeError("No trained main experiment")
        else:
            ensure_cache(INDEX_PATH, CACHE)
            if args.smoke:
                smoke(index, scalers)
                return
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            counts.to_csv(OUTPUT / "cohort_counts.csv", index=False)
            (OUTPUT / "source_records").mkdir(exist_ok=True)
            for target in TARGETS:
                shutil.copy2(BASELINE / f"task_records/{target}.csv", OUTPUT / f"source_records/{target}.csv")
            shutil.copy2(BASELINE / "target_scalers.json", OUTPUT / "target_scalers.json")
            contract = reference.sha256(path)
            jobs, rows = [], []
            for target in TARGETS:
                run = RESULTS / "runs" / target
                marker = run / "job_complete.json"
                if (marker.exists() and json.loads(marker.read_text()) == {"contract": contract}
                        and all((run / name).is_file() for name in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv"))):
                    rows.append({"architecture": config.ARCHITECTURE, "family": "regression", "target": target, "status": "ok"})
                else:
                    jobs.append({"architecture": config.ARCHITECTURE, "family": "regression", "target": target,
                                 "scaler": scalers[target], "weight_sha256": digest,
                                 "records_sha256": hashes[target], "contract": contract})
            workers = min(4, torch.cuda.device_count(), len(jobs))
            if jobs and not workers:
                raise RuntimeError("CUDA required")
            print(f"[scheduler] DINO main24h frame-loss regression head={HEAD_HIDDEN} pending={len(jobs)} gpus={workers}", flush=True)
            ctx = mp.get_context("spawn")
            if jobs:
                with ctx.Manager() as manager:
                    queue = manager.Queue()
                    for gpu in range(workers):
                        queue.put(gpu)
                    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker, initargs=(queue, HEAD_HIDDEN)) as pool:
                        futures = {pool.submit(worker, job): job for job in jobs}
                        for future in as_completed(futures):
                            target = futures[future]["target"]
                            try:
                                row = future.result()
                            except Exception:
                                row = {"architecture": config.ARCHITECTURE, "family": "regression", "target": target,
                                       "status": "failed", "error": traceback.format_exc()}
                                print(row["error"], flush=True)
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(OUTPUT / "run_index.csv", index=False)
                            print(f"[task-finished] regression/{target} {row['status']}", flush=True)
            pd.DataFrame(rows).to_csv(OUTPUT / "run_index.csv", index=False)
            if any(row["status"] != "ok" for row in rows):
                raise RuntimeError("DINO regression jobs failed; see run_index.csv")
            for name in ("metrics", "history"):
                pd.concat([pd.read_csv(RESULTS / f"runs/{target}/{name}.csv") for target in TARGETS], ignore_index=True).to_csv(RESULTS / f"{name}_all.csv", index=False)
        # Refuse to compare against a reference changed during the run.
        _, _, current, _, _ = preflight()
        if current != hashes or any(reference.sha256(BASELINE / f"runs/efficientnet_b0/{target}/video_predictions.csv") != value
                                    for target, value in manifest["baseline_predictions_sha256"].items()):
            raise RuntimeError("Baseline changed during DINO training")
        from .plots import plot_main_regression
        if head32_root is not None and any(reference.sha256(head32_root / f"runs/{target}/video_predictions.csv") != value
                                           for target, value in manifest["head32_baseline_predictions_sha256"].items()):
            raise RuntimeError("Head32 baseline changed during head64 training")
        plot_main_regression(RESULTS, BASELINE, TARGETS, head_hidden=HEAD_HIDDEN, head32_root=head32_root)
        (OUTPUT / "COMPLETE").write_text("ten 24h frame-loss DINO heads and paired main regression figures completed\n")
        print("[queue-complete] DINO main24h regression and paired comparison figures", flush=True)


if __name__ == "__main__":
    main()
