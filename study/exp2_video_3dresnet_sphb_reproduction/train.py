"""MSE training and video-level evaluation for the residual 3D CNN."""

import hashlib
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import (
    EARLY_STOPPING_VAL_MSE,
    EFFECTIVE_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    GRAD_CLIP_NORM,
    IMAGE_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    TRAIN_MICRO_BATCH_SIZE,
    WEIGHT_DECAY,
)
from .data import VideoClipDataset
from .models import PaperResidual3DRegressor, parameter_count


def _loader(records, index, train):
    dataset = VideoClipDataset(records, index)
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_MICRO_BATCH_SIZE if train else EVAL_BATCH_SIZE,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    return dataset, loader


def _prepare(video, device):
    # Dataset layout is B,T,C,H,W; Conv3D expects B,C,T,H,W.
    video = video.to(device, non_blocking=True).float().div_(255.0)
    batch, frames, channels, height, width = video.shape
    video = functional.interpolate(
        video.reshape(batch * frames, channels, height, width),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return video.reshape(batch, frames, channels, IMAGE_SIZE, IMAGE_SIZE).permute(
        0, 2, 1, 3, 4
    ).contiguous(memory_format=torch.channels_last_3d)


def _metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    return {
        "n": len(y_true),
        "mse_g_dl2": mse,
        "rmse_g_dl": float(np.sqrt(mse)),
        "mae_g_dl": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((y_true - y_true.mean()) ** 2)),
        "pearson_r": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
            else np.nan
        ),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    truths, predictions, indices = [], [], []
    for video, target, row_index in loader:
        video = _prepare(video, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(video)
        truths.extend(target.numpy().tolist())
        predictions.extend(prediction.float().cpu().numpy().tolist())
        indices.extend(row_index.numpy().tolist())
    return _metrics(truths, predictions), np.asarray(truths), np.asarray(predictions), indices


def _train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, sample_count, pending = 0.0, 0, 0
    started = time.perf_counter()
    for batch_index, (video, target, _) in enumerate(loader):
        video = _prepare(video, device)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(video)
            loss = functional.mse_loss(prediction, target)
        scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
        pending += 1
        final_batch = batch_index + 1 == len(loader)
        if pending == GRADIENT_ACCUMULATION_STEPS or final_batch:
            scaler.unscale_(optimizer)
            if pending != GRADIENT_ACCUMULATION_STEPS:
                correction = GRADIENT_ACCUMULATION_STEPS / pending
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        loss_sum += float(loss.detach().cpu()) * len(target)
        sample_count += len(target)
    torch.cuda.synchronize(device)
    return loss_sum / sample_count, time.perf_counter() - started


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(records, index, output_dir, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    splits = {
        name: records.loc[records["split"].eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    datasets, loaders = {}, {}
    for name in splits:
        datasets[name], loaders[name] = _loader(splits[name], index, name == "train")
    model = PaperResidual3DRegressor().to(device, memory_format=torch.channels_last_3d)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    best_mse, best_state, history = np.inf, None, []
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, seconds = _train_epoch(model, loaders["train"], optimizer, scaler, device)
        val_metrics, _, _, _ = evaluate(model, loaders["val"], device)
        history.append({
            "epoch": epoch,
            "train_mse_g_dl2": train_loss,
            "val_mse_g_dl2": val_metrics["mse_g_dl2"],
            "val_rmse_g_dl": val_metrics["rmse_g_dl"],
            "val_mae_g_dl": val_metrics["mae_g_dl"],
            "val_pearson_r": val_metrics["pearson_r"],
            "learning_rate": LEARNING_RATE,
            "train_seconds": seconds,
        })
        pd.DataFrame(history).to_csv(os.path.join(output_dir, "history.csv"), index=False)
        marker = ""
        if val_metrics["mse_g_dl2"] < best_mse:
            best_mse = val_metrics["mse_g_dl2"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = "*"
        print(
            f"[epoch] {epoch:03d}/{MAX_EPOCHS} train_MSE={train_loss:.4f} "
            f"val_MSE={val_metrics['mse_g_dl2']:.4f} val_RMSE={val_metrics['rmse_g_dl']:.4f} "
            f"val_r={val_metrics['pearson_r']:.4f} seconds={seconds:.1f}{marker}",
            flush=True,
        )
        if val_metrics["mse_g_dl2"] < EARLY_STOPPING_VAL_MSE:
            print(f"[paper-early-stop] val_MSE<{EARLY_STOPPING_VAL_MSE}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No finite checkpoint was produced")
    model.load_state_dict(best_state)
    metric_rows, prediction_rows = [], []
    for split in ("train", "val", "test"):
        metrics, truth, prediction, indices = evaluate(model, loaders[split], device)
        metric_rows.append({"split": split, **metrics})
        selected = splits[split].iloc[indices].reset_index(drop=True)
        for row, actual, estimate in zip(selected.itertuples(index=False), truth, prediction):
            prediction_rows.append({
                "hospital_id": row.hospital_id, "video_id": row.video_id, "split": split,
                "y_true_g_dl": actual, "y_pred_g_dl": estimate,
                "residual_g_dl": estimate - actual,
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    pd.DataFrame(prediction_rows).to_csv(os.path.join(output_dir, "video_predictions.csv"), index=False)
    checkpoint_path = os.path.join(output_dir, "model.pt")
    torch.save({
        "model_state_dict": best_state,
        "model": "paper_residual_3d_cnn",
        "target": "hemoglobin_g_dl",
        "parameters": parameter_count(model),
        "paper_effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": TRAIN_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "best_validation_mse_g_dl2": best_mse,
    }, checkpoint_path)
    for dataset in datasets.values():
        dataset.close()
    return metrics, checkpoint_path, _sha256(checkpoint_path)
