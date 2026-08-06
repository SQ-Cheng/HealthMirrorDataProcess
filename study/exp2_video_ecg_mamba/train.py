"""Single-task video+ECG Mamba abnormal-score regression."""

import copy
import itertools
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import (
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_LEARNING_RATE,
    PREFETCH_FACTOR,
    SCORE_TRANSFORM,
    SMOOTH_L1_BETA,
    TRAIN_BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    WEIGHT_DECAY,
)
from .data import VideoEcgWindowDataset, collate_windows
from .models import build_model, dependency_versions, parameter_counts


def _log(message):
    print(message, flush=True)


def _seed_worker(worker_id):
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def _loader(windows, training, seed, persistent=True):
    workers = TRAIN_NUM_WORKERS if training else EVAL_NUM_WORKERS
    if not persistent:
        workers = 0
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        VideoEcgWindowDataset(windows, training=training),
        batch_size=TRAIN_BATCH_SIZE if training else EVAL_BATCH_SIZE,
        shuffle=training,
        num_workers=workers,
        collate_fn=collate_windows,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0 and persistent,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )


def _to_device(batch, device):
    result = dict(batch)
    for key in (
        "frames",
        "frame_times",
        "frame_lengths",
        "ecg",
        "ecg_times",
        "ecg_lengths",
        "targets",
    ):
        result[key] = result[key].to(device, non_blocking=True)
    return result


def _forward(model, batch):
    return model(
        frames=batch["frames"],
        frame_times=batch["frame_times"],
        frame_lengths=batch["frame_lengths"],
        ecg=batch["ecg"],
        ecg_times=batch["ecg_times"],
        ecg_lengths=batch["ecg_lengths"],
    )


def _regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    valid = np.isfinite(targets) & np.isfinite(predictions)
    targets, predictions = targets[valid], predictions[valid]
    result = {
        "n_videos": int(len(targets)),
        "mae": np.nan,
        "rmse": np.nan,
        "median_ae": np.nan,
        "r2": np.nan,
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "sign_n": 0,
        "sign_balanced_accuracy": np.nan,
        "sign_roc_auc": np.nan,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }
    if not len(targets):
        return result
    result.update(
        {
            "mae": float(mean_absolute_error(targets, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
            "median_ae": float(median_absolute_error(targets, predictions)),
            "r2": (
                float(r2_score(targets, predictions))
                if len(targets) > 1 and np.var(targets) > 0
                else np.nan
            ),
            "pearson_r": (
                float(np.corrcoef(targets, predictions)[0, 1])
                if len(targets) > 1
                and np.std(targets) > 0
                and np.std(predictions) > 0
                else np.nan
            ),
            "spearman_r": (
                float(
                    pd.Series(targets).rank().corr(
                        pd.Series(predictions).rank(), method="pearson"
                    )
                )
                if len(targets) > 1
                else np.nan
            ),
        }
    )
    non_boundary = ~np.isclose(targets, 0.0, atol=1e-12)
    sign_targets = (targets[non_boundary] > 0).astype(np.uint8)
    sign_predictions = (predictions[non_boundary] > 0).astype(np.uint8)
    result["sign_n"] = int(len(sign_targets))
    if len(sign_targets) and len(np.unique(sign_targets)) == 2:
        tn, fp, fn, tp = confusion_matrix(
            sign_targets, sign_predictions, labels=[0, 1]
        ).ravel()
        result.update(
            {
                "sign_balanced_accuracy": float(
                    balanced_accuracy_score(sign_targets, sign_predictions)
                ),
                "sign_roc_auc": float(
                    roc_auc_score(sign_targets, predictions[non_boundary])
                ),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return result


def _video_metrics(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _regression_metrics([], []), frame
    score_spread = frame.groupby("video_id")["target_score"].agg(
        lambda values: float(values.max() - values.min())
    )
    if score_spread.gt(1e-6).any():
        raise ValueError("A video has inconsistent abnormal scores")
    video = (
        frame.groupby("video_id", as_index=False)
        .agg(
            hospital_id=("hospital_id", "first"),
            target_score=("target_score", "first"),
            prediction=("prediction", "mean"),
            window_count=("window_id", "count"),
        )
        .sort_values("video_id")
        .reset_index(drop=True)
    )
    metrics = _regression_metrics(
        video["target_score"].to_numpy(),
        video["prediction"].to_numpy(),
    )
    video["absolute_error"] = np.abs(
        video["prediction"] - video["target_score"]
    )
    video["target_abnormal"] = video["target_score"].gt(0).astype(np.uint8)
    video["predicted_abnormal"] = video["prediction"].gt(0).astype(np.uint8)
    return metrics, video


def _selection_score(metrics):
    if np.isfinite(metrics["mae"]):
        return -float(metrics["mae"])
    return -float(metrics["loss"])


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss, total_examples, optimizer_steps = 0.0, 0, 0
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batches = (
        loader
        if max_batches is None
        else itertools.islice(loader, int(max_batches))
    )
    for raw_batch in batches:
        batch = _to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            predictions = _forward(model, batch)
            loss = criterion(predictions, batch["targets"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {'train' if training else 'eval'} loss")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                optimizer_steps += 1
        batch_size = len(predictions)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        predictions_cpu = predictions.detach().float().cpu().numpy()
        targets_cpu = batch["targets"].detach().cpu().numpy()
        rows.extend(
            {
                "window_id": window_id,
                "video_id": video_id,
                "hospital_id": hospital_id,
                "target_score": float(target_score),
                "prediction": float(prediction),
            }
            for window_id, video_id, hospital_id, target_score, prediction in zip(
                raw_batch["window_ids"],
                raw_batch["video_ids"],
                raw_batch["hospital_ids"],
                targets_cpu,
                predictions_cpu,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    metrics, video_predictions = _video_metrics(rows)
    metrics.update(
        {
            "loss": total_loss / max(total_examples, 1),
            "windows": int(total_examples),
            "optimizer_steps": int(optimizer_steps),
            "seconds": elapsed,
            "windows_per_second": total_examples / max(elapsed, 1e-9),
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    return metrics, pd.DataFrame(rows), video_predictions


def _plot_history(history, path, target):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(frame.epoch, frame.train_loss, label="Train")
    axes[0].plot(frame.epoch, frame.val_loss, label="Validation")
    axes[0].set_ylabel(f"SmoothL1 loss (beta={SMOOTH_L1_BETA:g})")
    axes[1].plot(frame.epoch, frame.train_mae, label="Train MAE")
    axes[1].plot(frame.epoch, frame.val_mae, label="Validation MAE")
    axes[1].plot(frame.epoch, frame.val_rmse, label="Validation RMSE")
    axes[1].set_ylabel("Video-level error")
    axes[2].plot(frame.epoch, frame.train_pearson_r, label="Train Pearson r")
    axes[2].plot(frame.epoch, frame.val_pearson_r, label="Validation Pearson r")
    axes[2].plot(frame.epoch, frame.val_spearman_r, label="Validation Spearman r")
    axes[2].axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[2].set_ylim(-1.02, 1.02)
    axes[2].set_ylabel("Video-level correlation")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle(f"Raw video + 512 Hz ECG Mamba regression | {target}")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def train_task(
    target,
    windows_path,
    run_dir,
    device,
    seed,
    max_epochs=MAX_EPOCHS,
    patience_limit=EARLY_STOPPING_PATIENCE,
    max_batches=None,
):
    os.makedirs(run_dir, exist_ok=True)
    windows = pd.read_csv(
        windows_path,
        dtype={"hospital_id": str, "video_id": str, "window_id": str},
    )
    if not np.isfinite(pd.to_numeric(windows["target_score"], errors="coerce")).all():
        raise ValueError(f"Non-finite target scores for {target}")
    loaders = {
        split: _loader(
            windows.loc[windows["split"].eq(split)].reset_index(drop=True),
            training=split == "train",
            seed=seed + {"train": 0, "val": 1, "test": 2}[split],
            persistent=max_batches is None,
        )
        for split in ("train", "val", "test")
    }

    torch.cuda.set_device(device)
    model = build_model().to(device)
    counts = parameter_counts(model)
    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(int(max_epochs), 1), eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=1024.0
    )
    _log(
        f"[job-start] target={target} device={device} "
        f"train/val/test windows="
        f"{len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}/"
        f"{len(loaders['test'].dataset)} loss=SmoothL1(beta={SMOOTH_L1_BETA:g}) "
        f"parameters={counts['total']} mamba_parameters={counts['mamba']}"
    )

    history, best_state, best_score, best_epoch, patience = [], None, -np.inf, 0, 0
    started = time.perf_counter()
    for epoch in range(1, int(max_epochs) + 1):
        train_metrics, _, _ = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=max_batches,
        )
        with torch.no_grad():
            val_metrics, _, _ = _run_epoch(
                model,
                loaders["val"],
                criterion,
                device,
                max_batches=max_batches,
            )
        score = _selection_score(val_metrics)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            patience = 0
        else:
            patience += 1
        row = {
            "target": target,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_pearson_r": train_metrics["pearson_r"],
            "train_spearman_r": train_metrics["spearman_r"],
            "train_sign_bacc": train_metrics["sign_balanced_accuracy"],
            "train_sign_auc": train_metrics["sign_roc_auc"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_spearman_r": val_metrics["spearman_r"],
            "val_sign_bacc": val_metrics["sign_balanced_accuracy"],
            "val_sign_auc": val_metrics["sign_roc_auc"],
            "selection_score": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": train_metrics["optimizer_steps"],
            "train_windows_per_second": train_metrics["windows_per_second"],
            "peak_gpu_memory_gb": train_metrics["peak_gpu_memory_gb"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            os.path.join(run_dir, "history.csv"), index=False
        )
        _plot_history(history, os.path.join(run_dir, "history.png"), target)
        _log(
            f"[epoch] target={target} {epoch:03d}/{max_epochs}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_MAE={train_metrics['mae']:.4f} "
            f"train_r={train_metrics['pearson_r']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_MAE={val_metrics['mae']:.4f} "
            f"val_RMSE={val_metrics['rmse']:.4f} "
            f"val_r={val_metrics['pearson_r']:.4f} "
            f"val_sign_AUC={val_metrics['sign_roc_auc']:.4f} "
            f"steps={train_metrics['optimizer_steps']} "
            f"throughput={train_metrics['windows_per_second']:.2f}/s "
            f"peak_mem={train_metrics['peak_gpu_memory_gb']:.2f}GiB "
            f"patience={patience}/{patience_limit}{'*' if improved else ''}"
        )
        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        else:
            _log(f"[no-update] target={target} all optimizer steps were skipped")
        if patience >= int(patience_limit):
            _log(f"[early-stop] target={target} epoch={epoch}")
            break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint selected for {target}")
    model.load_state_dict(best_state)

    metrics_rows, segment_frames, video_frames = [], [], []
    for split in ("train", "val", "test"):
        evaluation_loader = (
            _loader(
                windows.loc[windows["split"].eq("train")].reset_index(drop=True),
                training=False,
                seed=seed + 10,
                persistent=max_batches is None,
            )
            if split == "train"
            else loaders[split]
        )
        with torch.no_grad():
            metrics, segments, videos = _run_epoch(
                model,
                evaluation_loader,
                criterion,
                device,
                max_batches=max_batches,
            )
        metrics_rows.append(
            {
                "target": target,
                "split": split,
                "selected_epoch": best_epoch,
                **metrics,
            }
        )
        segments.insert(0, "split", split)
        videos.insert(0, "split", split)
        segment_frames.append(segments)
        video_frames.append(videos)

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.concat(segment_frames, ignore_index=True).to_csv(
        os.path.join(run_dir, "segment_predictions.csv"), index=False
    )
    pd.concat(video_frames, ignore_index=True).to_csv(
        os.path.join(run_dir, "video_predictions.csv"), index=False
    )
    checkpoint = {
        "target": target,
        "task_type": "abnormal_score_regression",
        "score_transform": SCORE_TRANSFORM,
        "score_boundary": 0.0,
        "selected_epoch": best_epoch,
        "model_state_dict": best_state,
        "parameter_counts": counts,
        "dependencies": dependency_versions(),
    }
    torch.save(checkpoint, os.path.join(run_dir, "model.pt"))
    with open(
        os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "target": target,
                "task_type": "abnormal_score_regression",
                "seed": int(seed),
                "selected_epoch": int(best_epoch),
                "selection_metric": "negative_video_level_validation_mae",
                "selection_score": float(best_score),
                "loss": {
                    "name": "SmoothL1Loss",
                    "beta": SMOOTH_L1_BETA,
                    "auxiliary_classification_loss": False,
                },
                "score_transform": SCORE_TRANSFORM,
                "score_boundary": 0.0,
                "parameter_counts": counts,
                "dependencies": dependency_versions(),
            },
            handle,
            indent=2,
        )
    elapsed = time.perf_counter() - started
    test = metrics_frame.loc[metrics_frame["split"].eq("test")].iloc[0]
    _log(
        f"[job-done] target={target} selected_epoch={best_epoch} "
        f"test_MAE={test.mae:.4f} test_RMSE={test.rmse:.4f} "
        f"test_r={test.pearson_r:.4f} elapsed_min={elapsed / 60:.1f}"
    )
    return metrics_frame


__all__ = ["train_task"]
