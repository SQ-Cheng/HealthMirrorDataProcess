"""Rebuild and overwrite native224 Exp6 from the current complete lab table."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from study.common.run_video_loss_12h import Tee
from study.common.video_loss import DistinctLabViewBatchSampler
from study.exp2_face_history_head32_regression import source_data
from study.exp2_face_pretrained_head32_regression.frame_index import (
    FrameOffsetIndex, _index_is_reusable, build_or_reuse_frame_index,
)
from . import config
from .build_dataset import _pair_target, _add_split_and_scaling


OUTPUT = config.NATIVE_OUTPUT_DIR
LOGS = config.LOG_DIR / "face224"
PREVIEW = config.EXP_DIR / "cache/full24h_preparation"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def prepare(output, lab_pairs_per_batch=12):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "task_records").mkdir(exist_ok=True)
    digest = sha256(config.REPO_ROOT / "merged_lab_tests.csv")
    base, videos, quality = source_data.build_raw_video_source(
        str(output / "source_data"), config.NATIVE_TARGETS, 24,
    )
    source_data.validate_analyte_source_policies(quality, config.NATIVE_TARGETS)
    directories = [
        config.EXP_DIR.parent / "common/cache/face224_20frame_main",
        config.EXP_DIR.parent / "exp9_face_interpolated_lab_regression/cache/combined_frames20",
        config.EXP_DIR / "cache/full24h_frames20",
    ]
    index = index_path = None
    for directory in directories:
        if _index_is_reusable(directory, set(base.video_id), "20frame"):
            index_path = directory / "frame_offsets.npz"; index = FrameOffsetIndex.load(index_path)
            print(f"[frame-cache] reused {index_path}", flush=True)
            break
    if index is None:
        directory = directories[-1]
        index = build_or_reuse_frame_index(videos, str(directory), "20frame")
        index_path = directory / "frame_offsets.npz"
    if set(index.video_formats) != {"ffv1"}:
        raise RuntimeError("Only native224 FFV1 videos may be used")
    # Filter before selecting the closest video per lab event: an invalid crop
    # must not hide another valid video matched to the same report.
    usable = base.loc[base.video_id.isin(index.video_ids)].copy()
    base.loc[~base.video_id.isin(index.video_ids), ["hospital_id", "video_id"]].to_csv(
        output / "frame_exclusions.csv", index=False)
    records, scalers, selections, summaries, distributions = {}, {}, {}, [], []
    for target in config.NATIVE_TARGETS:
        paired, summary = _pair_target(usable, target, max_match_delta_hours=24)
        paired, scaler, audit, selection = _add_split_and_scaling(paired, target, config.SEED)
        paired.to_csv(output / f"task_records/{target}.csv", index=False)
        records[target] = paired; scalers[target] = scaler; selections[target] = selection
        distributions.extend(audit)
        summary.update(usable_pairs=len(paired), usable_patients=paired.hospital_id.nunique(),
                       usable_videos=pd.unique(pd.concat([paired.first_video_id, paired.second_video_id])).size)
        for split, group in paired.groupby("split"):
            summary.update({f"{split}_pairs": len(group), f"{split}_patients": group.hospital_id.nunique()})
        summaries.append(summary)
    if sha256(config.REPO_ROOT / "merged_lab_tests.csv") != digest:
        raise RuntimeError("Lab table changed during preparation")
    pd.DataFrame(summaries).to_csv(output / "task_summary.csv", index=False)
    pd.DataFrame(distributions).to_csv(output / "split_distribution.csv", index=False)
    (output / "target_scalers.json").write_text(json.dumps(scalers, indent=2) + "\n")
    manifest = {
        "experiment": "exp6_native224_full_data_24h_delta_regression", "targets": list(config.NATIVE_TARGETS),
        "lab_table_sha256": digest, "source_quality": quality,
        "frame_index": str(index_path), "frame_index_sha256": sha256(index_path),
        "matching_hours": 24, "matching": "same-admission nearest report to original session capture interval; validated Session Timestamp",
        "pairing": "consecutive unique lab events within patient; closest eligible native224 video per report; strictly increasing lab and video times",
        "native_frame_eligibility": "applied before choosing the closest video per laboratory event",
        "split_policy": "fresh 512-candidate patient-disjoint 60/20/20 distribution search on raw deltas",
        "split_selections": selections, "scaling": "train-only median/IQR of raw delta",
        "model_variant": "shared", "architecture": "shared ImageNet EfficientNet-B0; late-minus-early features; independent head32/model per analyte",
        "training": {
            "seed": config.SEED, "loss": "frame-pair weighted SmoothL1; unchanged inverse-patient-pair-count weights",
            "frames_per_video": 20, "views": list(config.VIEWS), "views_per_pair_per_batch": 1,
            "distinct_lab_delta_pairs_per_batch": lab_pairs_per_batch,
            "logical_frame_pairs_per_batch": lab_pairs_per_batch * 20,
            "face_images_per_full_batch": lab_pairs_per_batch * 40,
            "microbatch_frame_pairs": config.TRAIN_MICROBATCH_FRAME_PAIRS,
            "gradient_accumulation": "weighted sum over entire logical batch; one optimizer update and one gradient clip",
            "head_lr": config.HEAD_LEARNING_RATE, "finetune_lr": config.FINETUNE_LEARNING_RATE,
            "head_epochs": config.HEAD_MAX_EPOCHS, "finetune_epochs": config.FINETUNE_MAX_EPOCHS,
            "head_patience": config.HEAD_PATIENCE, "finetune_patience": config.FINETUNE_PATIENCE,
            "min_lr": config.MIN_LEARNING_RATE, "optimizer": "AdamW", "weight_decay": config.WEIGHT_DECAY,
            "scheduler": "cosine, no warmup", "compile": config.TORCH_COMPILE_MODE,
            "evaluation": "all 20 original-view frame pairs; mean prediction per video pair; raw measured delta is authoritative truth",
        },
        "records_sha256": {target: sha256(output / f"task_records/{target}.csv") for target in config.NATIVE_TARGETS},
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    preflight(output)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return records, scalers, index


def preflight(output):
    output = Path(output); manifest = json.loads((output / "experiment_manifest.json").read_text())
    if manifest["lab_table_sha256"] != sha256(config.REPO_ROOT / "merged_lab_tests.csv"):
        raise RuntimeError("Lab table differs from prepared data")
    index_path = Path(manifest["frame_index"]); index = FrameOffsetIndex.load(index_path)
    if sha256(index_path) != manifest["frame_index_sha256"] or not _index_is_reusable(index_path.parent, index.video_ids, "20frame"):
        raise RuntimeError("Native224 frame index is stale")
    scalers = json.loads((output / "target_scalers.json").read_text())
    from .run_all import _validate_records
    records = {}
    for target in config.NATIVE_TARGETS:
        path = output / f"task_records/{target}.csv"
        if sha256(path) != manifest["records_sha256"][target]: raise RuntimeError(f"Changed records: {target}")
        table = pd.read_csv(path, dtype={"hospital_id": str}, float_precision="round_trip")
        assert not table.pair_id.duplicated().any() and set(table.split) == {"train", "val", "test"}
        assert table.first_match_delta_h.between(0, 24).all() and table.second_match_delta_h.between(0, 24).all()
        train = table.loc[table.split.eq("train")].raw_delta.to_numpy(float)
        q1, median, q3 = np.quantile(train, [.25, .5, .75])
        assert scalers[target]["median"] == median and scalers[target]["iqr"] == q3 - q1
        np.testing.assert_allclose(table.scaled_delta, (table.raw_delta - median) / (q3 - q1), rtol=0, atol=1e-12)
        for column in ("first_video_id", "second_video_id"):
            assert table.groupby(["hospital_id", column]).split.nunique().le(1).all()
            assert all(index.frame_range(video)[1] - index.frame_range(video)[0] == 20 for video in table[column])
        training = table.loc[table.split.eq("train")].reset_index(drop=True)
        dataset = SimpleNamespace(expand_all_views=False, views=config.VIEWS,
                                  video_records=training.assign(clinical_event_id=training.pair_id),
                                  frame_video_rows=np.repeat(np.arange(len(training)), 20))
        batches = list(DistinctLabViewBatchSampler(dataset, manifest["training"]["logical_frame_pairs_per_batch"]))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(training) * 100))
        for batch in batches:
            groups = np.asarray(batch).reshape(-1, 20)
            assert len(set(groups[:, 0] // 100)) == len(groups)
            assert all(len(set(group % 5)) == 1 and len(set(group // 100)) == 1 for group in groups)
        records[target] = table
    _validate_records(records)
    print("[preflight-ok] eleven targets, latest labs, 24h, native224, patient-disjoint balanced split, train-only scaling, complete distinct-pair/view batches", flush=True)
    return records, scalers, index, manifest


def init_worker(index_path, queue):
    from . import run_all
    run_all._worker_init(index_path, queue)


def train_one(job):
    from .run_all import _worker_train
    run = Path(job["run_dir"]); run.mkdir(parents=True, exist_ok=True)
    with (run / "train.log").open("a", buffering=1) as log, \
         contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        result = _worker_train(job)
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    assert saved["target"] == job["target"] and saved["train_batch_policy"] == "distinct_lab_views"
    predictions = pd.read_csv(run / "pair_predictions.csv", dtype={"hospital_id": str}, float_precision="round_trip")
    records = pd.read_csv(job["records_path"], dtype={"hospital_id": str}, float_precision="round_trip")
    observed = predictions.set_index("pair_id").loc[records.pair_id]
    np.testing.assert_array_equal(observed.y_true, records.raw_delta)
    assert observed.frame_count.eq(20).all() and observed.split.to_numpy().tolist() == records.split.tolist()
    return result["metrics"]


def run(output):
    records, scalers, index, manifest = preflight(output)
    jobs = [{"target": target, "records_path": str(output / f"task_records/{target}.csv"),
             "run_dir": str(output / f"runs/{target}"), "seed": config.SEED, "scaler": scalers[target],
             "head_epochs": config.HEAD_MAX_EPOCHS, "finetune_epochs": config.FINETUNE_MAX_EPOCHS,
             "max_batches": None, "model_variant": "shared", "train_views": config.VIEWS,
             "training_options": {"train_batch_policy": "distinct_lab_views",
                                  "lab_pairs_per_batch": manifest["training"]["distinct_lab_delta_pairs_per_batch"],
                                  "microbatch_frame_pairs": config.TRAIN_MICROBATCH_FRAME_PAIRS}}
            for target in config.NATIVE_TARGETS]
    workers = min(4, torch.cuda.device_count(), len(jobs))
    if not workers: raise RuntimeError("CUDA is required")
    ctx = mp.get_context("spawn"); rows, metrics = [], []
    with ctx.Manager() as manager:
        queue = manager.Queue()
        for gpu in range(workers): queue.put(gpu)
        print(f"[scheduler] full-data Exp6 jobs={len(jobs)} GPUs={workers}", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                 initargs=(manifest["frame_index"], queue)) as pool:
            futures = {pool.submit(train_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    metrics.extend(future.result()); row = {"target": job["target"], "status": "ok"}
                except Exception:
                    row = {"target": job["target"], "status": "failed", "error": traceback.format_exc()}
                    print(row["error"], flush=True)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "run_index.csv", index=False)
                if metrics: pd.DataFrame(metrics).to_csv(output / "metrics_all.csv", index=False)
                print(f"[task-finished] {job['target']} {row['status']}", flush=True)
    if any(row["status"] != "ok" for row in rows): raise RuntimeError("Exp6 jobs failed; see run_index.csv")
    pd.concat([pd.read_csv(output / f"runs/{target}/history.csv") for target in config.NATIVE_TARGETS]).to_csv(
        output / "history_all.csv", index=False)
    from .plot_results import plot_results
    plot_results(output)
    preflight(output)
    (output / "COMPLETE").write_text("eleven native224 full-data 24h models and figures completed\n")


def smoke(output):
    from unittest.mock import patch
    from . import train
    records, scalers, index, _ = preflight(output)
    target = "hematocrit_low"
    source = records[target]
    subset = pd.concat([source.loc[source.split.eq("train")].head(12),
                        source.loc[source.split.ne("train")].groupby("split", group_keys=False).head(2)])
    with tempfile.TemporaryDirectory(prefix="exp6_twelve_pair_smoke_") as directory:
        with patch.object(train, "TORCH_COMPILE_ENABLED", False), \
             patch.object(train, "TRAIN_NUM_WORKERS", 0), patch.object(train, "EVAL_NUM_WORKERS", 0):
            train.train_task(target, subset, scalers[target], index, 0, directory, config.SEED,
                             head_epochs=1, finetune_epochs=1, max_batches=1,
                             train_batch_policy="distinct_lab_views", lab_pairs_per_batch=12,
                             microbatch_frame_pairs=config.TRAIN_MICROBATCH_FRAME_PAIRS)
        history = pd.read_csv(Path(directory) / "history.csv")
        assert history.train_model_inputs.eq(240).all() and np.isfinite(history.train_optimization_loss).all()
        checkpoint = torch.load(Path(directory) / "model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["lab_pairs_per_batch"] == 12 and checkpoint["microbatch_frame_pairs"] == 120
    torch.cuda.empty_cache()
    print("[smoke-ok] real native224 input; twelve observed delta pairs; weighted accumulation; two stages; checkpoint saved", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lab-pairs-per-batch", type=int, choices=(6, 12), default=12)
    args = parser.parse_args(argv)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / ".full_data.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            prepare(PREVIEW, args.lab_pairs_per_batch); return
        if OUTPUT.exists():
            if not args.overwrite: raise FileExistsError("Use --overwrite to replace the Exp6 main results")
            shutil.rmtree(OUTPUT)
        prepare(OUTPUT, args.lab_pairs_per_batch)
        smoke(OUTPUT)
        run(OUTPUT)


if __name__ == "__main__":
    main()
