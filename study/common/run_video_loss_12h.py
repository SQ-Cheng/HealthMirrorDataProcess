"""Run paired single-split 12h classification/regression video-loss ablations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import tempfile
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression import config
from study.exp2_face_pretrained_head32_regression.data import validate_source_data
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler
from .lab_run_version import RUN_TAG, versioned


STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY / "exp2_face_pretrained_head32_regression/outputs/ablations/lab_match_12h_face224"
INDEX_PATH = STUDY / "common/cache/face224_20frame/frame_offsets.npz"
if RUN_TAG:
    SOURCE = STUDY / "common/outputs" / RUN_TAG / "reference"
    INDEX_PATH = STUDY / "common/cache" / RUN_TAG / "frame_offsets.npz"
STATE = versioned(STUDY / "common/outputs/video_loss_12h")
OUTPUTS = {family: versioned(STUDY / f"exp2_face_pretrained_head32_{family}/outputs/ablations/lab_match_12h_face224") / "video_loss"
           for family in ("classification", "regression")}
GPU = INDEX = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_seed(family, target):
    if family == "classification":
        return config.SEED
    offset = int.from_bytes(hashlib.sha256(f"{config.SEED}:efficientnet_b0:{target}".encode()).digest()[:4], "little")
    return (config.SEED + offset) % (2**31 - 1)


def preflight():
    if not (SOURCE / ("DATA_COMPLETE" if RUN_TAG else "COMPLETE")).is_file():
        raise RuntimeError("The single-split 12h reference is incomplete")
    validate_source_data(SOURCE / "source_data", expected_max_delta_hours=12)
    index = FrameOffsetIndex.load(INDEX_PATH)
    if set(index.video_formats) != {"ffv1"} or not _index_is_reusable(INDEX_PATH.parent, index.video_ids, "20frame"):
        raise RuntimeError("Native224 index is stale")
    saved_manifest = json.loads((SOURCE / "experiment_manifest.json").read_text())
    if RUN_TAG and saved_manifest["lab_table_sha256"] != sha256(STUDY.parent / "merged_lab_tests.csv"):
        raise RuntimeError("Lab table changed after cohort preparation")
    if saved_manifest["frame_index_sha256"] != sha256(INDEX_PATH):
        raise RuntimeError("Reference frame index changed")
    scalers = json.loads((SOURCE / "target_scalers.json").read_text())["targets"]
    hashes, counts = {}, []
    for target in config.TARGETS:
        path = SOURCE / f"task_records/{target}.csv"
        records = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
        if (records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().max() != 1
                or set(records.split) != {"train", "val", "test"}
                or not records.match_delta_h.between(0, 12 + 1e-9).all()
                or not np.isfinite(records[["raw_value", "robust_scaled_raw_value"]]).all().all()):
            raise RuntimeError(f"Invalid source records: {target}")
        fitted = fit_robust_target_scaler(target, records, config.SCORE_DEFINITIONS[target]["unit"])
        if fitted.to_dict() != scalers[target]:
            raise RuntimeError(f"Reference scaler is not train-only: {target}")
        np.testing.assert_array_equal(fitted.transform(records.raw_value).astype(np.float32),
                                      records.robust_scaled_raw_value.to_numpy(np.float32))
        if any(index.frame_range(video)[1] - index.frame_range(video)[0] != 20 for video in records.video_id):
            raise RuntimeError(f"Incomplete frame coverage: {target}")
        hashes[target] = sha256(path)
        for split, group in records.groupby("split"):
            if group.binary_label.nunique() != 2:
                raise RuntimeError(f"Single-class split: {target}/{split}")
            counts.append({"target": target, "split": split, "videos": len(group),
                           "patients": group.hospital_id.nunique(), "positive_videos": int(group.binary_label.sum())})
    print("[preflight-ok] shared single split, 12h, native224, 20frames, 5views", flush=True)
    return index, scalers, hashes, pd.DataFrame(counts)


def init_worker(queue):
    global GPU, INDEX
    GPU = int(queue.get())
    torch.cuda.set_device(GPU)
    INDEX = FrameOffsetIndex.load(INDEX_PATH)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def train_one(job):
    family, target = job["family"], job["target"]
    root = Path(job.get("output", OUTPUTS[family]))
    batch_policy = job.get("batch_policy", "chunked")
    run = root / f"runs/efficientnet_b0/{target}"
    run.mkdir(parents=True, exist_ok=True)
    seed = job_seed(family, target)
    path = root / f"task_records/{target}.csv"
    with (run / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        if family == "classification":
            from study.exp2_binary_classification_common.engine import train_task
            train_task("face_only", target, GPU, seed, output_dir=root,
                       records_path=path, reference_records_path=SOURCE / f"task_records/{target}.csv",
                       frame_index_path=INDEX_PATH, loss_level="video",
                       train_batch_policy=batch_policy)
        else:
            from study.exp2_face_pretrained_head32_regression.train import train_task
            torch.set_num_threads(1)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            train_task("efficientnet_b0", target, INDEX,
                       pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}),
                       RobustTargetScaler(**job["scaler"]), config.WEIGHTS_DIR, str(run), loss_level="video",
                       train_batch_policy=batch_policy)
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["loss_level"] != "video" or checkpoint["train_batch_policy"] != batch_policy:
        raise RuntimeError(f"Wrong saved protocol: {run}")
    prediction = pd.read_csv(run / "video_predictions.csv", dtype={"hospital_id": str, "video_id": str})
    source = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
    prediction = prediction.sort_values("video_id").reset_index(drop=True)
    source = source.sort_values("video_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(prediction[["hospital_id", "video_id", "split"]],
                                  source[["hospital_id", "video_id", "split"]], check_dtype=False)
    count_column = "input_count" if family == "classification" else "frame_count"
    if not prediction[count_column].eq(20).all():
        raise RuntimeError("Every evaluated video must contain exactly twenty frames")
    expected = source.binary_label if family == "classification" else source.raw_value
    np.testing.assert_allclose(prediction.y_true, expected, rtol=0, atol=1e-7)
    (run / "job_complete.json").write_text(json.dumps({"contract": job["contract"], "seed": seed}))
    del checkpoint
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return {"family": family, "target": target, "architecture": "efficientnet_b0", "status": "ok", "seed": seed}


def prepare(scalers, hashes, counts, prepared_records=None):
    from study.exp2_face_pretrained_head32_regression.models import WEIGHT_FILES
    contracts = {}
    for family, root in OUTPUTS.items():
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "family": family, "matching_hours": 12, "source_resolution": 224,
            "source": str(SOURCE), "targets": list(config.TARGETS), "source_records_sha256": hashes,
            "frame_index_sha256": sha256(INDEX_PATH), "architecture": "efficientnet_b0",
            "pretrained_weight_sha256": sha256(Path(config.WEIGHTS_DIR) / WEIGHT_FILES["efficientnet_b0"]),
            "head_hidden_features": 32, "loss_level": "video", "frames_per_video": 20,
            "training_views": list(config.VIEW_NAMES), "train_batch_policy": "chunked",
            "frame_batch_size": 240, "video_view_groups_per_batch": 12,
            "aggregation": "mean of twenty frame probabilities" if family == "classification" else "mean of twenty scaled frame predictions",
            "augmentation_loss_policy": "one video-level loss per view; all five views retained",
            "head_lr": config.HEAD_LEARNING_RATE, "head_epochs": config.HEAD_MAX_EPOCHS,
            "head_patience": config.HEAD_PATIENCE, "finetune_lr": config.FINETUNE_LEARNING_RATE,
            "finetune_epochs": config.FINETUNE_MAX_EPOCHS, "finetune_patience": config.FINETUNE_PATIENCE,
            "min_lr": config.MIN_LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
            "freeze_bn_stats": False, "warmup_epochs": 0,
            "compile": config.TORCH_COMPILE_ENABLED, "compile_mode": config.TORCH_COMPILE_MODE,
            "job_seeds": {target: job_seed(family, target) for target in config.TARGETS},
            "split_policy": "exact single patient-disjoint 12h regression reference; no new seed search",
            "selection_metric": "validation bACC" if family == "classification" else "validation raw-unit MAE",
            "baseline": None if family == "classification" or RUN_TAG else str(SOURCE),
            "comparison_note": "Refreshed cohort: no matched frame-loss baseline was retrained" if RUN_TAG else ("No single-split 12h frame-loss classifier exists; five-fold results are not a paired baseline" if family == "classification" else "paired against existing 12h frame-loss regression"),
        }
        if prepared_records is not None:
            manifest.update(
                train_batch_policy="distinct_lab_views", loss_unit="video_view",
                batch_unique_measurements=12, views_per_video_per_batch=1,
                clinical_event_id_policy="canonical hospital_id + target-specific matched lab report timestamp",
                clinical_event_metadata_sha256=sha256(SOURCE / "source_data/base_manifest.csv"),
                prepared_records_sha256={target: hashlib.sha256(text.encode()).hexdigest()
                                         for target, text in prepared_records.items()},
            )
        path = root / "experiment_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"Existing video-loss contract differs: {root}")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        contracts[family] = sha256(path)
        (root / "task_records").mkdir(exist_ok=True)
        for target in config.TARGETS:
            if prepared_records is None:
                shutil.copy2(SOURCE / f"task_records/{target}.csv", root / f"task_records/{target}.csv")
            else:
                (root / f"task_records/{target}.csv").write_text(prepared_records[target])
        shutil.copy2(SOURCE / "target_scalers.json", root / "target_scalers.json")
        counts.to_csv(root / "cohort_counts.csv", index=False)
    return contracts


def finalize(family):
    root = OUTPUTS[family]
    runs = [root / f"runs/efficientnet_b0/{target}" for target in config.TARGETS]
    for name in ("metrics", "history"):
        pd.concat([pd.read_csv(run / f"{name}.csv") for run in runs], ignore_index=True).to_csv(root / f"{name}_all.csv", index=False)
    if family == "classification":
        from .selected_5fold_plots import plot_fold_classification
        plot_fold_classification(root)
        import matplotlib.pyplot as plt
        from .plot_layout import target_grid_shape, target_grid_figsize
        from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS
        rows, cols = target_grid_shape(len(config.TARGETS))
        figure, axes = plt.subplots(rows, cols, figsize=target_grid_figsize(rows, cols), squeeze=False)
        history = pd.read_csv(root / "history_all.csv")
        for axis, target in zip(axes.flat, config.TARGETS):
            selected = history.loc[history.target.eq(target)]
            axis.plot(selected.global_epoch, selected.train_loss, label="Train", color="#2878B5")
            axis.plot(selected.global_epoch, selected.val_loss, label="Validation", color="#CB6547")
            axis.set(title=TASK_LABELS[target], xlabel="Epoch",
                     ylabel="View-level weighted BCE" if root.name == "view_loss_12_distinct_labs" else "Video-level weighted BCE")
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(root / "figures/training_history.png", dpi=180)
        plt.close(figure)
        (root / "COMPLETE").write_text("training and figures completed\n")
    else:
        from study.exp2_face_pretrained_head32_regression.plot_results import main as plot
        from study.exp2_face_pretrained_head32_regression.plot_patient_diverse_comparison import plot_comparison
        plot(root)
        (root / "COMPLETE").write_text("training and figures completed\n")
        if RUN_TAG:
            return
        try:
            plot_comparison(SOURCE, root, reference_label="Frame-level loss",
                            candidate_label="Video-level loss", figure_prefix="frame_vs_video_loss")
        except Exception:
            (root / "COMPLETE").unlink()
            raise


def smoke(index, scalers):
    target = "hemoglobin_low"
    records = pd.read_csv(SOURCE / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
    subset = pd.concat([group.groupby("binary_label", group_keys=False).head(1)
                        for _, group in records.groupby("split")], ignore_index=True)
    from study.exp2_binary_classification_common.engine import train_task as classify
    from study.exp2_face_pretrained_head32_regression import train as regress
    with tempfile.TemporaryDirectory(prefix="video_loss_smoke_") as name:
        root = Path(name)
        path = root / "records.csv"
        subset.to_csv(path, index=False)
        classify("face_only", target, 0, config.SEED, smoke=True, output_dir=root / "classification",
                 records_path=path, reference_records_path=SOURCE / f"task_records/{target}.csv",
                 frame_index_path=INDEX_PATH, loss_level="video")
        with patch.object(regress, "TORCH_COMPILE_ENABLED", False):
            regress.train_task("efficientnet_b0", target, index, subset, RobustTargetScaler(**scalers[target]),
                               config.WEIGHTS_DIR, str(root / "regression"), head_epochs=1, finetune_epochs=1,
                               max_batches=1, loss_level="video")
        for family in ("classification", "regression"):
            run = root / (f"classification/runs/efficientnet_b0/{target}" if family == "classification" else "regression")
            saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
            if saved["loss_level"] != "video":
                raise AssertionError("Smoke checkpoint has the wrong objective")
            history = pd.read_csv(run / "history.csv")
            if not history.train_model_inputs.eq(200).all() or not np.isfinite(history[["train_loss", "val_loss"]]).all().all():
                raise AssertionError("Smoke coverage or loss failed")
    print("[smoke-ok] both objectives: two stages, real FFV1 frames, finite losses, saved checkpoints", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    index, scalers, hashes, counts = preflight()
    if args.check_only:
        print(counts.to_string(index=False))
        return
    if args.smoke:
        smoke(index, scalers)
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contracts = prepare(scalers, hashes, counts)
        jobs, rows = [], []
        for target in config.TARGETS:
            for family, root in OUTPUTS.items():
                run = root / f"runs/efficientnet_b0/{target}"
                marker = run / "job_complete.json"
                expected = {"contract": contracts[family], "seed": job_seed(family, target)}
                if marker.exists() and json.loads(marker.read_text()) == expected and all(
                    (run / f).is_file() for f in ("model.pt", "metrics.csv", "history.csv", "video_predictions.csv")
                ):
                    rows.append({"family": family, "target": target, "architecture": "efficientnet_b0", "status": "ok"})
                else:
                    jobs.append({"family": family, "target": target, "scaler": scalers[target], "contract": contracts[family]})
        workers = min(4, torch.cuda.device_count(), len(jobs))
        if jobs and workers < 1:
            raise RuntimeError("CUDA is required")
        print(f"[scheduler] video-loss 12h tasks=16 pending={len(jobs)} gpus={workers}", flush=True)
        ctx = mp.get_context("spawn")
        if jobs:
            with ctx.Manager() as manager:
                queue = manager.Queue()
                for gpu in range(workers):
                    queue.put(gpu)
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker, initargs=(queue,)) as pool:
                    futures = {pool.submit(train_one, job): job for job in jobs}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            row = future.result()
                        except Exception:
                            row = {"family": job["family"], "target": job["target"], "architecture": "efficientnet_b0",
                                   "status": "failed", "error": traceback.format_exc()}
                            print(row["error"], flush=True)
                        rows.append(row)
                        pd.DataFrame([r for r in rows if r["family"] == job["family"]]).to_csv(OUTPUTS[job["family"]] / "run_index.csv", index=False)
                        print(f"[task-finished] {job['family']}/{job['target']} {row['status']}", flush=True)
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Video-loss jobs failed; completed tasks can be reused")
        for family, root in OUTPUTS.items():
            pd.DataFrame([r for r in rows if r["family"] == family]).to_csv(root / "run_index.csv", index=False)
            finalize(family)
        if preflight()[2] != hashes:
            raise RuntimeError("Source labels changed during training")
        (STATE / "COMPLETE").write_text("both single-split video-loss experiments completed\n")
        print("[queue-complete] classification/regression video-loss 12h", flush=True)


if __name__ == "__main__":
    main()
