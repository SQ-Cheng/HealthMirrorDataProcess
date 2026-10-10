"""Fit independent concat/gated regression heads on identical DINO main data."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.common.run_video_loss_12h import Tee, sha256
from study.exp2_face_architecture_ablation.train import train_task
from study.exp2_face_dinov3_frozen.data import DinoFeatureDataset, no_feature_fitting
from study.exp2_face_dinov3_frozen.features import ensure_cache as ensure_dino
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from . import config
from .features import ensure_cache
from .models import build_model


INDEX = DEVICE = None


class DualDataset(DinoFeatureDataset):
    def __init__(self, index, records, views):
        super().__init__(index, records, views, "regression", config.DINO_CACHE)
        self.farl = np.load(config.CACHE / "features.npy", mmap_mode="r")
        if self.features.shape != (len(index.starts), 5, 384) or self.farl.shape != (len(index.starts), 5, 512):
            raise RuntimeError("Encoder cache alignment/dimensions are invalid")

    def __getitem__(self, sample):
        frame, view = divmod(sample, len(self.views))
        video = self.frame_video_rows[frame]
        index = self.frame_indices[frame]
        features = np.concatenate((self.features[index, view], self.farl[index, view]))
        return torch.from_numpy(features), torch.tensor(self.labels_by_video[video]), torch.tensor(frame), torch.tensor(view)


def loader(index, records, architecture, family, train):
    dataset = DualDataset(index, records, config.reference.config.VIEWS if train else ("original",))
    sampler = DistinctLabViewBatchSampler(dataset, 240) if train else VideoViewBatchSampler(dataset, 500, False)
    return dataset, DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())


def training_config(output=config.OUTPUT, epochs=80):
    return SimpleNamespace(HERE=config.HERE, OUTPUT_DIR=output, reference=config.reference.reference, LOSS_LEVEL="frame",
                           LEARNING_RATES={variant: 2e-4 for variant in config.VARIANTS},
                           MIN_LEARNING_RATES={variant: 1e-6 for variant in config.VARIANTS},
                           MAX_EPOCHS=epochs, PATIENCE=12, WEIGHT_DECAY=.001)


def init_worker(queue):
    global INDEX, DEVICE
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")
    INDEX = FrameOffsetIndex.load(config.INDEX_PATH)


def worker(job):
    root = config.OUTPUT / f"regression/{job['architecture']}/runs/{job['target']}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_task(job, INDEX, DEVICE, experiment_config=training_config(), model_factory=build_model,
                   loader_factory=loader, feature_scaler=no_feature_fitting)
    saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    saved.update(encoders_frozen=True, dino_weight_sha256=job["dino_weight_sha256"], farl_weight_sha256=job["farl_weight_sha256"],
                 feature_dimensions=[384, 512], projection_dimension=64 if job["architecture"] == "gated" else None,
                 head_hidden=32, matching_hours=24, clinical_source_sha256=job["source_sha256"], experiment_contract=job["contract"])
    torch.save(saved, root / "model.pt")
    (root / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}) + "\n")
    torch.cuda.empty_cache()
    return {"target": job["target"], "architecture": job["architecture"], "status": "ok"}


def smoke(index, scalers):
    target = "lactate_high"
    records = pd.read_csv(config.reference.BASELINE / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
    training = records.loc[records.split.eq("train")].drop_duplicates("clinical_event_id").groupby("binary_label", group_keys=False).head(6)
    held = records.loc[records.split.ne("train")].groupby("split", group_keys=False).head(2)
    subset = pd.concat([training, held], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="dual_encoder_smoke_") as name:
        from pathlib import Path
        root = Path(name)
        (root / "source_records").mkdir()
        subset.to_csv(root / f"source_records/{target}.csv", index=False)
        for variant in config.VARIANTS:
            train_task({"target": target, "architecture": variant, "family": "regression", "scaler": scalers[target]},
                       index, torch.device("cuda:0"), experiment_config=training_config(root, 1),
                       model_factory=build_model, loader_factory=loader, feature_scaler=no_feature_fitting)
            checkpoint = torch.load(root / f"regression/{variant}/runs/{target}/model.pt", map_location="cpu", weights_only=True)
            assert checkpoint["parameters"] == sum(p.numel() for p in build_model(variant).parameters())
    print("[smoke-ok] both fusion heads, real shared frame/view features, loss, optimizer and checkpoints", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    index, scalers, hashes, counts, lab_hash = config.reference.preflight()
    dino_output = config.reference.config.HERE / "outputs/main24h_frame_loss"
    dino_manifest = json.loads((dino_output / "experiment_manifest.json").read_text())
    if not (dino_output / "COMPLETE").is_file() or dino_manifest["source_records_sha256"] != hashes:
        raise RuntimeError("Completed DINO comparator data do not match")
    if args.check_only:
        print(counts.to_string(index=False))
        return
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_dino(config.INDEX_PATH, config.DINO_CACHE)
        ensure_cache()
        if args.smoke:
            smoke(index, scalers)
            return
        manifest = {"targets": list(config.TARGETS), "source_records_sha256": hashes, "lab_table_sha256": lab_hash,
                    "frame_index_sha256": sha256(config.INDEX_PATH), "scalers_sha256": sha256(config.reference.BASELINE / "target_scalers.json"),
                    "dino_weight_sha256": dino_manifest["backbone_weight_sha256"], "farl_weight_sha256": sha256(config.WEIGHTS),
                    "clip_revision": config.CLIP_REVISION, "encoders_frozen": True, "variants": list(config.VARIANTS),
                    "feature_dimensions": [384, 512], "projection_dimension": 64, "head_hidden": 32,
                    "parameters": {variant: sum(p.numel() for p in build_model(variant).parameters()) for variant in config.VARIANTS},
                    "matching_hours": 24, "frames_per_video": 20, "views": list(config.reference.config.VIEWS),
                    "batch_distinct_labs": 12, "batch_frames": 240, "loss_level": "frame",
                    "loss": "unweighted SmoothL1 beta=.5 on train-only robust raw values", "max_epochs": 80,
                    "patience": 12, "learning_rate": 2e-4, "min_lr": 1e-6, "weight_decay": .001,
                    "dropout": .25, "optimizer": "AdamW", "scheduler": "cosine; no warmup", "compile": False,
                    "normalization": {"dino": "original ImageNet normalization", "farl_mean": list(config.CLIP_MEAN), "farl_std": list(config.CLIP_STD)},
                    "dino_comparator_manifest_sha256": sha256(dino_output / "experiment_manifest.json")}
        path = config.OUTPUT / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Existing fusion contract changed; results are not overwritten")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        contract = sha256(path)
        counts.to_csv(config.OUTPUT / "cohort_counts.csv", index=False)
        (config.OUTPUT / "source_records").mkdir(exist_ok=True)
        for target in config.TARGETS:
            shutil.copy2(config.reference.BASELINE / f"task_records/{target}.csv", config.OUTPUT / f"source_records/{target}.csv")
        shutil.copy2(config.reference.BASELINE / "target_scalers.json", config.OUTPUT / "target_scalers.json")
        jobs, rows = [], []
        for target in config.TARGETS:
            for variant in config.VARIANTS:
                root = config.OUTPUT / f"regression/{variant}/runs/{target}"
                marker = root / "job_complete.json"
                if marker.exists() and json.loads(marker.read_text()) == {"contract": contract} and all((root / f).is_file() for f in ("model.pt", "history.csv", "metrics.csv", "video_predictions.csv")):
                    rows.append({"target": target, "architecture": variant, "status": "ok"})
                else:
                    jobs.append({"target": target, "architecture": variant, "family": "regression", "scaler": scalers[target],
                                 "contract": contract, "dino_weight_sha256": manifest["dino_weight_sha256"],
                                 "farl_weight_sha256": manifest["farl_weight_sha256"], "source_sha256": hashes[target]})
        workers = min(4, torch.cuda.device_count(), len(jobs))
        context = mp.get_context("spawn")
        print(f"[scheduler] fusion heads={len(jobs)} GPUs={workers}", flush=True)
        if jobs:
            if not workers:
                raise RuntimeError("CUDA required")
            with context.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(queue,)) as pool:
                    futures = {pool.submit(worker, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            row = future.result()
                        except Exception:
                            row = {"target": job["target"], "architecture": job["architecture"], "status": "failed", "error": traceback.format_exc()}
                            print(row["error"], flush=True)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(config.OUTPUT / "run_index.csv", index=False)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Fusion jobs failed; see run_index.csv")
        for variant in config.VARIANTS:
            root = config.OUTPUT / f"regression/{variant}"
            for field in ("metrics", "history"):
                pd.concat([pd.read_csv(root / f"runs/{target}/{field}.csv") for target in config.TARGETS], ignore_index=True).to_csv(root / f"{field}_all.csv", index=False)
        if config.reference.preflight()[2] != hashes or sha256(dino_output / "experiment_manifest.json") != manifest["dino_comparator_manifest_sha256"]:
            raise RuntimeError("Clinical/DINO reference changed during training")
        from .plots import plot_results
        plot_results()
        (config.OUTPUT / "COMPLETE").write_text("20 independent frozen dual-encoder regressors and paired figures completed\n")
        print("[queue-complete] DINOv3+FaRL regression", flush=True)


if __name__ == "__main__":
    main()
