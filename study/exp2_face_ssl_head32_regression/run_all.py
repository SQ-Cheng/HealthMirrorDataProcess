"""Global patient split, all-frame train-only BYOL, then ten independent regressors."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.data import (
    _distribution_audit, _plot_split_distributions, validate_source_data,
)
from study.exp2_face_pretrained_head32_regression.frame_index import (
    FrameOffsetIndex, build_or_reuse_frame_index, _index_is_reusable,
)
from study.exp2_face_pretrained_head32_regression.scaling import (
    fit_robust_target_scaler, RobustTargetScaler, write_target_scalers,
)
from study.exp2_face_shared_backbone_head32_regression.run_all import select_shared_split, sha256
from study.common.run_video_loss_12h import Tee, job_seed
from . import config
from .byol import pretrain, DiverseVideoBatchSampler


GPU = INDEX = None


def run_ssl_subprocess(extra=()):
    gpus = min(4, torch.cuda.device_count())
    if not gpus: raise RuntimeError("BYOL requires at least one GPU")
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = "GNU"
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    subprocess.run([sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node", str(gpus), "-m",
                    "study.exp2_face_ssl_head32_regression.ssl_worker", *extra],
                   cwd=config.ROOT, env=environment, check=True)


def prepare():
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT / "task_records").mkdir(exist_ok=True)
    if (config.OUTPUT / "ssl/encoder.pt").exists(): raise FileExistsError("Do not rebuild a completed SSL protocol")
    reference = json.loads((config.BASE / "direct_field_extension_protocol.json").read_text())
    if sha256(config.ROOT / "merged_lab_tests.csv") != reference["lab_table_sha256"]:
        raise RuntimeError("Current lab table differs from the main reference")
    validate_source_data(config.BASE / "source_data", 24)
    expected = {**reference["original_eight_records_sha256"], **reference["record_hashes"][str(config.BASE)]}
    records, hashes = {}, {}
    for target in config.TARGETS:
        path = config.BASE / f"task_records/{target}.csv"; hashes[target] = sha256(path)
        if hashes[target] != expected[target]: raise RuntimeError(f"Main task changed: {target}")
        records[target] = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
    assignment, selection = select_shared_split(records)
    scalers, summaries, audits, pairs = {}, [], [], []
    for target, table in records.items():
        table["original_main_split"] = table.split; table["split"] = table.hospital_id.map(assignment)
        scaler = fit_robust_target_scaler(target, table, config.reference.SCORE_DEFINITIONS[target]["unit"])
        table["robust_scaled_raw_value"] = scaler.transform(table.raw_value); scalers[target] = scaler
        table.to_csv(config.OUTPUT / f"task_records/{target}.csv", index=False)
        audit, pair = _distribution_audit(table, target); audits.extend(audit); pairs.extend(pair)
        row = {"target": target, "videos": len(table), "patients": table.hospital_id.nunique()}
        for split, group in table.groupby("split"):
            row.update({f"{split}_videos": len(group), f"{split}_patients": group.hospital_id.nunique()})
        summaries.append(row)
    pd.DataFrame({"hospital_id": assignment.keys(), "split": assignment.values()}).to_csv(config.OUTPUT / "patient_split.csv", index=False)
    write_target_scalers(scalers, config.OUTPUT / "target_scalers.json")
    pd.DataFrame(summaries).to_csv(config.OUTPUT / "task_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(config.OUTPUT / "split_distribution_audit.csv", index=False)
    pd.DataFrame(pairs).to_csv(config.OUTPUT / "split_distribution_pairwise.csv", index=False)
    _plot_split_distributions(records, config.OUTPUT)
    audit_path = config.BASE / "source_data/raw_video_audit.csv"
    inventory = pd.read_csv(audit_path, dtype={"hospital_id": str, "video_id": str})
    valid = inventory.status.isin(("retained_24h_pool", "supported_lab_outside_24h", "patient_without_supported_lab"))
    inventory["split"] = inventory.hospital_id.map(assignment)
    candidates = inventory.loc[valid & inventory.split.eq("train")].drop_duplicates("video_id").copy()
    parsed = candidates.video_id.str.extract(r"^(mirror\d+)_patient_(\d+)$")
    if parsed.isna().any().any(): raise RuntimeError("Invalid video identities")
    candidates["mirror"] = parsed[0]; candidates["lab_patient_id"] = parsed[1].astype(int)
    candidates = candidates[["hospital_id", "video_id", "mirror", "lab_patient_id", "split"]]
    ssl_index = build_or_reuse_frame_index(candidates, str(config.CACHE / "ssl_allframes"), "allframes")
    ssl_videos = candidates.loc[candidates.video_id.isin(ssl_index.video_ids)].sort_values("video_id").reset_index(drop=True)
    ssl_videos.to_csv(config.OUTPUT / "ssl_train_videos.csv", index=False)
    candidates.loc[~candidates.video_id.isin(ssl_index.video_ids)].to_csv(config.OUTPUT / "ssl_video_exclusions.csv", index=False)
    inventory[["hospital_id", "video_id", "status", "split"]].to_csv(config.OUTPUT / "ssl_inventory_audit.csv", index=False)
    source_frames = sum(ssl_index.frame_range(video)[1] - ssl_index.frame_range(video)[0] for video in ssl_videos.video_id)
    manifest = {
        "experiment": "exp2_face_ssl_head32_regression", "method": "BYOL", "seed": config.SEED,
        "targets": list(config.TARGETS), "lab_table_sha256": reference["lab_table_sha256"],
        "source": str(config.BASE), "source_records_sha256": hashes, "raw_video_audit_sha256": sha256(audit_path),
        "split_selection": selection, "split": "one global patient split across SSL and all supervised tasks",
        "supervised_frame_index": reference["frame_index"], "supervised_frame_index_sha256": reference["frame_index_sha256"],
        "ssl_frame_index": str(config.CACHE / "ssl_allframes/frame_offsets.npz"),
        "ssl_frame_index_sha256": sha256(config.CACHE / "ssl_allframes/frame_offsets.npz"),
        "ssl_video_records_sha256": sha256(config.OUTPUT / "ssl_train_videos.csv"),
        "ssl": {
            "train_videos": len(ssl_videos), "train_patients": ssl_videos.hospital_id.nunique(), "accepted_frames": source_frames,
            "anchors_per_epoch": source_frames * len(config.VIEWS), "epochs": config.SSL_EPOCHS,
            "views": list(config.VIEWS), "positive_pairs": "same accepted frame, two different existing view types",
            "frame_policy": "all accepted native224 frames of validated sessions assigned to training patients, including outside24h/unlabelled sessions",
            "patient_policy": "never include validation, test, unknown or unassigned patients",
            "labels": "no clinical values, targets or timestamps provided to the BYOL learner",
            "batch": "64 anchors per GPU, four-process DDP; global video-diverse batches; all real frame/views once; empty tail ranks use loss-zero train-only forward slots",
            "initialization": "local ImageNet EfficientNet-B0", "projector_predictor_hidden": config.SSL_MLP_HIDDEN,
            "projection_dim": config.SSL_PROJECTION_DIM, "backbone_lr": config.SSL_BACKBONE_LR,
            "mlp_lr": config.SSL_MLP_LR, "weight_decay": config.SSL_WEIGHT_DECAY,
            "warmup_epochs": config.SSL_WARMUP_EPOCHS, "scheduler": "step-wise linear warmup then cosine",
            "ema": "0.996 toward 1, parameters only, update after successful optimizer steps",
            "teacher_bn": "training-batch statistics, independently updated buffers; no held-out forward",
            "loss": "mean of two stop-gradient normalized-MSE directions; no negatives or clinical labels",
            "checkpoint_selection": "fixed final epoch, never use validation/test images for SSL selection",
            "export": "online features only; discard projector, predictor and EMA teacher",
        },
        "downstream": {
            "tasks": "independent EfficientNet-B0/backbone/head32 for every analyte; identical initial BYOL encoder file",
            "hours": 24, "frames_per_video": 20, "views": list(config.VIEWS), "loss": "frame-level unweighted SmoothL1(beta=0.5)",
            "scaling": "per-task training-only median/IQR", "batch": "12 different matched assays, one view and twenty frames each",
            "head_lr": config.reference.HEAD_LEARNING_RATE, "fine_lr": config.reference.FINETUNE_LEARNING_RATE,
            "head_epochs": config.reference.HEAD_MAX_EPOCHS, "fine_epochs": config.reference.FINETUNE_MAX_EPOCHS,
            "head_patience": config.reference.HEAD_PATIENCE, "fine_patience": config.reference.FINETUNE_PATIENCE,
            "min_lr": config.reference.MIN_LEARNING_RATE, "weight_decay": config.reference.WEIGHT_DECAY,
            "compile": "Inductor dynamic, CUDA Graphs disabled to avoid the previously observed OOM",
        },
        "comparison": "original-main patient splits differ; full cohort and common held-out test comparisons reported separately",
        "record_hashes": {target: sha256(config.OUTPUT / f"task_records/{target}.csv") for target in config.TARGETS},
        "references": ["https://arxiv.org/abs/2006.07733",
                       "https://github.com/google-deepmind/deepmind-research/tree/master/byol"],
    }
    (config.OUTPUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    preflight()
    print(f"[prepared] SSL videos={len(ssl_videos)} frames={source_frames} five-view anchors/epoch={source_frames*5}", flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def preflight():
    manifest = json.loads((config.OUTPUT / "experiment_manifest.json").read_text())
    assert manifest["seed"] == config.SEED
    assert manifest["ssl"]["epochs"] == config.SSL_EPOCHS and manifest["ssl"]["views"] == list(config.VIEWS)
    if sha256(config.ROOT / "merged_lab_tests.csv") != manifest["lab_table_sha256"]:
        raise RuntimeError("Current lab table changed")
    assert sha256(config.BASE / "source_data/raw_video_audit.csv") == manifest["raw_video_audit_sha256"]
    assignment = pd.read_csv(config.OUTPUT / "patient_split.csv", dtype={"hospital_id": str})
    if assignment.hospital_id.duplicated().any(): raise RuntimeError("Ambiguous common patient split")
    mapping = assignment.set_index("hospital_id").split
    videos = pd.read_csv(config.OUTPUT / "ssl_train_videos.csv", dtype={"hospital_id": str, "video_id": str})
    assert not videos.video_id.duplicated().any()
    assert set(videos.split) == {"train"} and videos.hospital_id.map(mapping).eq("train").all()
    assert not set(videos.hospital_id) & set(assignment.loc[assignment.split.ne("train"), "hospital_id"])
    assert sha256(config.OUTPUT / "ssl_train_videos.csv") == manifest["ssl_video_records_sha256"]
    assert not {"raw_value", "binary_label", "label_time_unix"} & set(videos.columns)
    indexes = []
    for prefix, policy in (("ssl", "allframes"), ("supervised", "20frame")):
        path = Path(manifest[f"{prefix}_frame_index"]); index = FrameOffsetIndex.load(path)
        assert sha256(path) == manifest[f"{prefix}_frame_index_sha256"]
        assert set(index.video_formats) == {"ffv1"} and _index_is_reusable(path.parent, index.video_ids, policy)
        indexes.append(index)
    assert set(videos.video_id).issubset(indexes[0].video_lookup)
    scalers = json.loads((config.OUTPUT / "target_scalers.json").read_text())["targets"]
    records = {}
    for target in config.TARGETS:
        assert sha256(config.BASE / f"task_records/{target}.csv") == manifest["source_records_sha256"][target]
        path = config.OUTPUT / f"task_records/{target}.csv"
        assert sha256(path) == manifest["record_hashes"][target]
        table = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
        assert table.split.eq(table.hospital_id.map(mapping)).all() and set(table.split) == {"train", "val", "test"}
        assert table.match_delta_h.between(0, 24 + 1e-9).all() and not table.video_id.duplicated().any()
        assert all(indexes[1].frame_range(video)[1] - indexes[1].frame_range(video)[0] == 20 for video in table.video_id)
        assert fit_robust_target_scaler(target, table, config.reference.SCORE_DEFINITIONS[target]["unit"]).to_dict() == scalers[target]
        records[target] = table
    print("[preflight-ok] common patient split; SSL train-only all-frame pool; ten unchanged 24h tasks; no cross-task patient leakage", flush=True)
    return records, videos, indexes[0], indexes[1], scalers, manifest


def safe_execution(model, architecture, target, stage, device):
    if device.type != "cuda": return model, "eager"
    return torch.compile(model, dynamic=True, options={"triton.cudagraphs": False}), "inductor:dynamic_no_cudagraphs"


def init_worker(queue, index_path):
    global GPU, INDEX
    GPU = int(queue.get()); torch.cuda.set_device(GPU); torch.set_num_threads(1)
    INDEX = FrameOffsetIndex.load(index_path)


def train_one(target, scaler, encoder, contract):
    from study.exp2_face_pretrained_head32_regression import train
    run = config.OUTPUT / f"runs/efficientnet_b0/{target}"; run.mkdir(parents=True, exist_ok=True)
    seed = job_seed("regression", target)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    records = pd.read_csv(config.OUTPUT / f"task_records/{target}.csv", dtype={"hospital_id": str}, float_precision="round_trip")
    with (run / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), \
         patch.object(train, "_execution_model", safe_execution):
        train.train_task("efficientnet_b0", target, INDEX, records, RobustTargetScaler(**scaler), config.reference.WEIGHTS_DIR,
                         str(run), initial_encoder_state_path=str(encoder), train_batch_policy="distinct_lab_views", loss_level="frame")
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    assert saved["target"] == target and saved["initial_encoder_sha256"] == sha256(encoder)
    (run / "job_complete.json").write_text(json.dumps({"contract": contract, "target": target}) + "\n")
    torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    return {"architecture": "efficientnet_b0", "target": target, "status": "ok", "seed": seed}


def downstream():
    records, videos, ssl_index, index, scalers, manifest = preflight()
    encoder = config.OUTPUT / "ssl/encoder.pt"
    saved = torch.load(encoder, map_location="cpu", weights_only=True)
    assert set(saved["train_patients"]) == set(videos.hospital_id)
    assert saved["contract"] == sha256(config.OUTPUT / "experiment_manifest.json")
    if saved["ssl_epochs"] != config.SSL_EPOCHS:
        transition = json.loads(encoder.with_name("transition.json").read_text())
        assert saved.get("early_stopped_by_user") and transition["contract"] == saved["contract"]
        assert transition["completed_ssl_epochs"] == saved["ssl_epochs"]
        assert transition["encoder_sha256"] == sha256(encoder)
    print(f"[downstream-start] initialize all ten independent models from BYOL epoch={saved['ssl_epochs']} "
          f"encoder_sha256={sha256(encoder)}; no further SSL training", flush=True)
    contract = sha256(config.OUTPUT / "experiment_manifest.json") + ":" + sha256(encoder)
    jobs, rows = [], []
    for target in config.TARGETS:
        run = config.OUTPUT / f"runs/efficientnet_b0/{target}"
        marker = run / "job_complete.json"
        if marker.exists() and json.loads(marker.read_text()).get("contract") != contract:
            raise RuntimeError(f"Different existing run contract; not overwriting: {run}")
        if marker.exists() and all((run / name).exists() for name in ("model.pt", "history.csv", "metrics.csv", "video_predictions.csv")):
            rows.append({"architecture": "efficientnet_b0", "target": target, "status": "ok"})
        else: jobs.append(target)
    if jobs:
        workers = min(4, torch.cuda.device_count(), len(jobs)); ctx = mp.get_context("spawn")
        if not workers: raise RuntimeError("CUDA is required")
        with ctx.Manager() as manager:
            queue = manager.Queue()
            for gpu in range(workers): queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                     initargs=(queue, manifest["supervised_frame_index"])) as pool:
                futures = {pool.submit(train_one, target, scalers[target], encoder, contract): target for target in jobs}
                for future in as_completed(futures):
                    try: row = future.result()
                    except Exception as error:
                        row = {"architecture": "efficientnet_b0", "target": futures[future], "status": "failed", "error": repr(error)}
                    rows.append(row); pd.DataFrame(rows).to_csv(config.OUTPUT / "run_index.csv", index=False)
                    print(f"[task-finished] {row}", flush=True)
    if any(row["status"] != "ok" for row in rows): raise RuntimeError("Supervised BYOL-transfer jobs failed")
    pd.DataFrame(rows).to_csv(config.OUTPUT / "run_index.csv", index=False)
    for name in ("history", "metrics"):
        pd.concat([pd.read_csv(config.OUTPUT / f"runs/efficientnet_b0/{target}/{name}.csv")
                   for target in config.TARGETS]).to_csv(config.OUTPUT / f"{name}_all.csv", index=False)
    from study.exp2_face_pretrained_head32_regression import plot_results
    with patch.object(plot_results, "EXPERIMENT_LABEL", "BYOL-initialized independent face regressors"):
        plot_results.main(config.OUTPUT)
    from .plots import plot_all
    plot_all()
    preflight()
    (config.OUTPUT / "COMPLETE").write_text("train-only BYOL and ten independent two-stage regressors/figures completed\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=("all", "ssl", "downstream"), default="all")
    args = parser.parse_args()
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.reuse_prepared: prepare()
        if args.prepare_only: return
        records, videos, ssl_index, supervised_index, scalers, manifest = preflight()
        if args.smoke:
            from .smoke import run_smoke
            run_smoke(records, videos, ssl_index, supervised_index, scalers); return
        if args.phase in ("all", "ssl"):
            # Isolate CUDA/DataParallel memory from the subsequent four independent workers.
            run_ssl_subprocess()
        if args.phase in ("all", "downstream"): downstream()


if __name__ == "__main__":
    main()
