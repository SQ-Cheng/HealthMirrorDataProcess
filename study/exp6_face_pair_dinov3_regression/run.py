"""Audit the main Exp6 cohort, cache frozen features, fit 22 independent heads."""

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

from study.common.run_video_loss_12h import Tee
from study.exp2_face_dinov3_frozen.backbone import ensure_source, weight_sha256
from study.exp2_face_dinov3_frozen.features import ensure_cache
from study.exp6_face_pair_lab_delta.run_full_data import preflight as baseline_preflight, sha256
from study.exp6_face_pair_lab_delta.train import _prepare
from study.exp2_face_pretrained_head32_regression.train import _prepare_images

from . import config
from .models import head_parameters
from .train import train_task


INDEX = DEVICE = None


def preflight():
    if not (config.BASELINE / "COMPLETE").is_file():
        raise RuntimeError("Exp6 main regression must be complete before comparison")
    records, scalers, index, baseline = baseline_preflight(config.BASELINE)
    training = baseline["training"]
    expected = {"frames_per_video": 20, "views": list(config.VIEWS),
                "views_per_pair_per_batch": 1, "distinct_lab_delta_pairs_per_batch": 12,
                "logical_frame_pairs_per_batch": 240, "microbatch_frame_pairs": 120}
    if baseline["model_variant"] != "shared" or any(training.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Unexpected Exp6 main model/batch protocol")
    counts = []
    for target, table in records.items():
        prediction = pd.read_csv(config.BASELINE / f"runs/{target}/pair_predictions.csv",
                                 dtype={"hospital_id": str}, float_precision="round_trip")
        prediction = prediction.sort_values("pair_id").reset_index(drop=True)
        original = table.sort_values("pair_id").reset_index(drop=True)
        keys = ["hospital_id", "pair_id", "split", "first_video_id", "second_video_id"]
        pd.testing.assert_frame_equal(prediction[keys], original[keys], check_dtype=False)
        np.testing.assert_array_equal(prediction.y_true, original.raw_delta)
        if not prediction.frame_count.eq(20).all():
            raise RuntimeError("Baseline used a different frame-pair count")
        for split, group in table.groupby("split"):
            counts.append({"target": target, "split": split, "pairs": len(group),
                           "patients": group.hospital_id.nunique(),
                           "videos": pd.unique(pd.concat([group.first_video_id, group.second_video_id])).size})
    return records, scalers, index, baseline, pd.DataFrame(counts)


def manifest_for(baseline):
    return {
        "experiment": "exp6_frozen_dinov3_pair_delta_regression", "baseline": str(config.BASELINE),
        "baseline_manifest_sha256": sha256(config.BASELINE / "experiment_manifest.json"),
        "lab_table_sha256": baseline["lab_table_sha256"], "records_sha256": baseline["records_sha256"],
        "scalers_sha256": sha256(config.BASELINE / "target_scalers.json"),
        "baseline_predictions_sha256": {target: sha256(config.BASELINE / f"runs/{target}/pair_predictions.csv")
                                          for target in baseline["targets"]},
        "frame_index": baseline["frame_index"], "frame_index_sha256": baseline["frame_index_sha256"],
        "matching_hours": 24, "targets": baseline["targets"], "head_widths": list(config.HEAD_WIDTHS),
        "backbone": "official DINOv3 ViT-S/16 LVD1689M", "backbone_frozen": True,
        "backbone_weight_sha256": weight_sha256(), "backbone_source_revision": config.dino.REPO_REVISION,
        "backbone_parameters": 21601152, "head_parameters": {str(width): head_parameters(width) for width in config.HEAD_WIDTHS},
        "fusion": "same frozen encoder for both timepoints; late normalized CLS minus early normalized CLS",
        "loss": "per-frame-pair SmoothL1(beta=0.5), inverse patient pair count weighting exactly as main Exp6",
        "scaling": "exact main train-only median/IQR of raw second-minus-first laboratory delta",
        "frame_pairs_per_batch": 240, "distinct_lab_pairs_per_batch": 12, "microbatch_frame_pairs": 120,
        "gradient_accumulation": "same weighted logical-batch mean and one optimizer step/clip",
        "frames_per_video": 20, "views": list(config.VIEWS), "views_per_pair_per_batch": 1,
        "epoch_coverage": "each frame-pair/view once; no dropping or oversampling",
        "evaluation": "20 original-view aligned frame pairs; mean scaled predictions, inverse scaling; authoritative raw delta",
        "seed": config.SEED, "optimizer": "AdamW", "learning_rate": config.LEARNING_RATE,
        "minimum_lr": config.MIN_LEARNING_RATE, "max_epochs": config.MAX_EPOCHS, "patience": config.PATIENCE,
        "weight_decay": config.WEIGHT_DECAY, "dropout": config.dino.DROPOUT, "gradient_clip": config.GRAD_CLIP,
        "scheduler": "cosine; no warmup", "stages": 1, "selection": "validation raw-unit pair MAE",
        "mixed_precision": "float16 autocast with GradScaler", "compile": False,
        "comparison_scope": "identical clinical data, split, frame/view alignment, weighting, loss, and evaluation; frozen DINO head-only versus two-stage fine-tuned EN-B0, not a backbone-only ablation",
    }


def init_worker(queue, index_path):
    global INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
    INDEX = FrameOffsetIndex.load(index_path)


def worker(job):
    root = config.OUTPUT / f"head{job['hidden']}/runs/{job['target']}"
    root.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(config.OUTPUT / f"task_records/{job['target']}.csv",
                          dtype={"hospital_id": str}, float_precision="round_trip")
    with (root / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_task(job["target"], job["hidden"], records, job["scaler"], INDEX, DEVICE, root, job["provenance"])
    saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if saved["head_parameters"] != head_parameters(job["hidden"]) or not saved["backbone_frozen"]:
        raise RuntimeError("Wrong DINO head checkpoint")
    (root / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}) + "\n")
    gc.collect()
    torch.cuda.empty_cache()
    return {"target": job["target"], "head_hidden": job["hidden"], "status": "ok"}


def smoke(records, scalers, index, manifest):
    images = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    codes = torch.arange(5).repeat(2, 1)
    torch.testing.assert_close(_prepare(images, codes, torch.device("cuda:0")),
                               _prepare_images(images, codes, "bicubic", torch.device("cuda:0")), rtol=0, atol=0)
    target = "hemoglobin_low"
    table = records[target]
    subset = pd.concat([table.loc[table.split.eq("train")].head(12),
                        table.loc[table.split.ne("train")].groupby("split", group_keys=False).head(2)], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="exp6_dino_smoke_") as name:
        for width in config.HEAD_WIDTHS:
            path = Path(name) / f"head{width}"
            train_task(target, width, subset, scalers[target], index, torch.device("cuda:0"), path,
                       {"backbone_weight_sha256": manifest["backbone_weight_sha256"]}, epochs=1)
            saved = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            history = pd.read_csv(path / "history.csv")
            assert saved["head_parameters"] == head_parameters(width)
            assert history.train_model_inputs.iloc[0] == 1200 and history.optimizer_steps.iloc[0] == 5
    print("[smoke-ok] both heads; exact five-view pixel preprocessing; real weighted pair losses/checkpoints/evaluation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    ensure_source()
    records, scalers, index, baseline, counts = preflight()
    manifest = manifest_for(baseline)
    if args.check_only:
        print(counts.to_string(index=False))
        return
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = config.OUTPUT / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("DINO experiment contract changed; existing results are not overwritten")
        if args.plot_only:
            if not path.exists():
                raise RuntimeError("No DINO experiment results")
        else:
            reuse = (config.HERE.parent / "common/cache/face224_20frame_main/frame_offsets.npz",
                     config.dino.HERE / "cache/main24h_frame_loss")
            ensure_cache(Path(baseline["frame_index"]), config.CACHE, reuse_from=reuse)
            if args.smoke:
                smoke(records, scalers, index, manifest)
                return
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            (config.OUTPUT / "task_records").mkdir(exist_ok=True)
            for target in records:
                shutil.copy2(config.BASELINE / f"task_records/{target}.csv", config.OUTPUT / f"task_records/{target}.csv")
            shutil.copy2(config.BASELINE / "target_scalers.json", config.OUTPUT / "target_scalers.json")
            counts.to_csv(config.OUTPUT / "cohort_counts.csv", index=False)
            contract = sha256(path)
            jobs, rows = [], []
            for target in records:
                for width in config.HEAD_WIDTHS:
                    root = config.OUTPUT / f"head{width}/runs/{target}"
                    marker = root / "job_complete.json"
                    if (marker.exists() and json.loads(marker.read_text()) == {"contract": contract}
                            and all((root / file).is_file() for file in ("model.pt", "history.csv", "metrics.csv", "pair_predictions.csv"))):
                        rows.append({"target": target, "head_hidden": width, "status": "ok"})
                    else:
                        jobs.append({"target": target, "hidden": width, "scaler": scalers[target], "contract": contract,
                                     "provenance": {"backbone_weight_sha256": manifest["backbone_weight_sha256"],
                                                    "backbone_source_revision": config.dino.REPO_REVISION,
                                                    "source_records_sha256": baseline["records_sha256"][target],
                                                    "frame_index_sha256": baseline["frame_index_sha256"], "experiment_contract": contract}})
            workers = min(4, torch.cuda.device_count(), len(jobs))
            if jobs and not workers:
                raise RuntimeError("CUDA required")
            print(f"[scheduler] Exp6 DINO regression jobs={len(jobs)} GPUs={workers} heads=32,64", flush=True)
            ctx = mp.get_context("spawn")
            if jobs:
                with ctx.Manager() as manager:
                    queue = manager.Queue()
                    for gpu in range(workers):
                        queue.put(gpu)
                    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                             initargs=(queue, baseline["frame_index"])) as pool:
                        futures = {pool.submit(worker, job): job for job in jobs}
                        for future in as_completed(futures):
                            job = futures[future]
                            try:
                                row = future.result()
                            except Exception:
                                row = {"target": job["target"], "head_hidden": job["hidden"], "status": "failed", "error": traceback.format_exc()}
                                print(row["error"], flush=True)
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(config.OUTPUT / "run_index.csv", index=False)
                            print(f"[task-finished] head{job['hidden']}/{job['target']} {row['status']}", flush=True)
            pd.DataFrame(rows).to_csv(config.OUTPUT / "run_index.csv", index=False)
            if any(row["status"] != "ok" for row in rows):
                raise RuntimeError("DINO jobs failed; see run_index.csv")
            for width in config.HEAD_WIDTHS:
                root = config.OUTPUT / f"head{width}"
                for name in ("metrics", "history"):
                    pd.concat([pd.read_csv(root / f"runs/{target}/{name}.csv") for target in records], ignore_index=True).to_csv(root / f"{name}_all.csv", index=False)
        if manifest_for(preflight()[3]) != manifest:
            raise RuntimeError("Main Exp6 reference changed during training")
        from .plots import plot_results
        plot_results()
        (config.OUTPUT / "COMPLETE").write_text("22 frozen DINO heads and three-model paired comparisons completed\n")
        print("[queue-complete] Exp6 DINO head32/head64 regression and comparison figures", flush=True)


if __name__ == "__main__":
    main()
