"""Two-stage EfficientNet recovery training with video-level evaluation."""

from functools import lru_cache
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import (
    CROP_SCALE,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    FINETUNE_BACKBONE_LEARNING_RATE,
    FINETUNE_HEAD_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    GRAD_CLIP_NORM,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_LEARNING_RATE,
    PREFETCH_FACTOR,
    SMOOTH_L1_BETA,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZE,
    TRAIN_VIEWS,
    WEIGHT_DECAY,
)
from .data import RecoveryFrameDataset
from .models import build_model, freeze_backbone, parameter_counts, unfreeze_last_stage


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(index, records, train):
    dataset = RecoveryFrameDataset(
        index,
        records,
        views=TRAIN_VIEWS if train else ("original",),
        expand_views=train,
    )
    workers = TRAIN_NUM_WORKERS if train else EVAL_NUM_WORKERS
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_SOURCE_BATCH_SIZE if train else EVAL_BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
        drop_last=False,
    )
    return dataset, loader


def _prepare_images(images, view_codes, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    view_codes = view_codes.to(device, non_blocking=True)
    if view_codes.ndim == 2:
        images = images.repeat_interleave(view_codes.shape[1], dim=0)
        view_codes = view_codes.flatten()
    flip = view_codes.eq(1)
    if flip.any():
        images[flip] = torch.flip(images[flip], dims=(-1,))
    crop = view_codes.eq(2)
    output = torch.empty((len(images), 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    regular = ~crop
    if regular.any():
        output[regular] = F.interpolate(
            images[regular], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    if crop.any():
        height, width = images.shape[-2:]
        crop_height, crop_width = int(round(height * CROP_SCALE)), int(round(width * CROP_SCALE))
        top, left = (height - crop_height) // 2, (width - crop_width) // 2
        output[crop] = F.interpolate(
            images[crop, :, top:top + crop_height, left:left + crop_width],
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    mean = output.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = output.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((output - mean) / std).contiguous(memory_format=torch.channels_last)


def _weighted_loss(predictions, labels, weights):
    losses = F.smooth_l1_loss(
        predictions, labels, beta=SMOOTH_L1_BETA, reduction="none"
    )
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def _train_epoch(model, loader, optimizer, scaler, device, encoder_frozen):
    if encoder_frozen:
        model.eval()
        getattr(model, "_orig_mod", model).classifier.train()
    else:
        model.train()
    total, weight_total, model_inputs = 0.0, 0.0, 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for images, labels, _, view_codes, weights in loader:
        images = _prepare_images(images, view_codes, device)
        repeat = view_codes.shape[1]
        labels = labels.repeat_interleave(repeat).to(device, non_blocking=True)
        weights = weights.repeat_interleave(repeat).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(images).squeeze(1)
            loss = _weighted_loss(predictions, labels, weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer); scaler.update()
        batch_weight = float(weights.sum().detach().cpu())
        total += float(loss.detach().cpu()) * batch_weight
        weight_total += batch_weight
        model_inputs += len(labels)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss": total / max(weight_total, 1e-8),
        "model_inputs": model_inputs,
        "seconds": elapsed,
        "throughput": model_inputs / max(elapsed, 1e-8),
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


@torch.no_grad()
def _evaluate_frames(model, loader, device):
    model.eval(); predictions, labels, frame_rows = [], [], []
    for images, target, rows, view_codes, _ in loader:
        images = _prepare_images(images, view_codes, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(images).squeeze(1)
        predictions.append(prediction.float().cpu().numpy())
        labels.append(target.numpy())
        frame_rows.append(rows.numpy())
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(frame_rows)


def _metrics(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    absolute = np.abs(prediction - truth)
    smooth_l1 = np.where(
        absolute < SMOOTH_L1_BETA,
        0.5 * absolute**2 / SMOOTH_L1_BETA,
        absolute - 0.5 * SMOOTH_L1_BETA,
    )
    return {
        "n": len(truth),
        "loss": float(smooth_l1.mean()),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "explained_variance": (
            float(explained_variance_score(truth, prediction)) if len(truth) > 1 else np.nan
        ),
        "pearson_r": (
            float(np.corrcoef(truth, prediction)[0, 1])
            if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0 else np.nan
        ),
        "spearman_r": (
            float(pd.Series(truth).rank().corr(pd.Series(prediction).rank()))
            if len(truth) > 1 else np.nan
        ),
    }


def _video_evaluation(model, dataset, loader, device, split):
    truth, prediction, frame_rows = _evaluate_frames(model, loader, device)
    video_rows = dataset.frame_video_rows[frame_rows]
    aggregated = pd.DataFrame({
        "video_row": video_rows,
        "y_true": truth,
        "y_pred": prediction,
    }).groupby("video_row", as_index=False).agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        frame_count=("y_pred", "size"),
        frame_prediction_std=("y_pred", "std"),
    )
    info = dataset.records.iloc[aggregated.video_row.to_numpy(int)][
        ["hospital_id", "video_id", "recovery_score", "hours_after_surgery",
         "postoperative_duration_hours"]
    ].reset_index(drop=True)
    result = pd.concat([info, aggregated.drop(columns="video_row")], axis=1)
    result.insert(0, "split", split)
    if not np.allclose(result.recovery_score, result.y_true, atol=1e-7):
        raise AssertionError(f"Aggregated labels differ from records for {split}")
    return _metrics(result.y_true, result.y_pred), result


def _clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _compile(model, stage):
    if not TORCH_COMPILE_ENABLED:
        return model, "eager"
    torch._dynamo.config.suppress_errors = True
    compiled = torch.compile(model, mode=TORCH_COMPILE_MODE, fullgraph=False, dynamic=False)
    print(f"[compile] stage={stage} mode={TORCH_COMPILE_MODE}", flush=True)
    return compiled, f"torch_compile:{TORCH_COMPILE_MODE}"


def _plot_history(history, path, seed):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = (("loss", "SmoothL1 loss"), ("mae", "Video MAE"),
               ("r2", "Video R²"), ("pearson_r", "Video Pearson r"))
    for axis, (metric, title) in zip(axes.flat, metrics):
        for stage, group in frame.groupby("stage", sort=False):
            axis.plot(group.global_epoch, group[f"train_{metric}"], label=f"{stage} train")
            axis.plot(group.global_epoch, group[f"val_{metric}"], linestyle="--", label=f"{stage} val")
        axis.set_title(title); axis.set_xlabel("Epoch"); axis.grid(alpha=0.2); axis.legend(fontsize=8)
    figure.suptitle(f"Exp4 postoperative recovery | seed {seed}")
    figure.tight_layout(); figure.savefig(path, dpi=180, bbox_inches="tight"); plt.close(figure)


def _run_stage(stage, model, head, datasets, loaders, device, run_dir, history,
               max_epochs, patience_limit, optimizer):
    execution_model, backend = _compile(model, stage)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=MIN_LEARNING_RATE)
    best_state, best_mae, patience = None, np.inf, 0
    start_epoch = len(history)
    for stage_epoch in range(1, max_epochs + 1):
        train_step = _train_epoch(
            execution_model, loaders["train_augmented"], optimizer, scaler,
            device, encoder_frozen=stage == "head",
        )
        train_metrics, _ = _video_evaluation(execution_model, datasets["train"], loaders["train"], device, "train")
        val_metrics, _ = _video_evaluation(execution_model, datasets["val"], loaders["val"], device, "val")
        row = {
            "stage": stage, "stage_epoch": stage_epoch, "global_epoch": start_epoch + stage_epoch,
            "train_optimization_loss": train_step["loss"],
            "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items() if key not in {"n", "loss"}},
            **{f"val_{key}": value for key, value in val_metrics.items() if key not in {"n", "loss"}},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_model_inputs": train_step["model_inputs"],
            "train_seconds": train_step["seconds"],
            "train_inputs_per_second": train_step["throughput"],
            "peak_gpu_memory_gb": train_step["peak_memory_gb"],
            "execution_backend": backend,
        }
        history.append(row); pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_metrics["mae"] < best_mae - 1e-4:
            best_mae, best_state, patience, marker = val_metrics["mae"], _clone_state(model), 0, "*"
        else:
            patience += 1; marker = ""
        print(
            f"[epoch] stage={stage} {stage_epoch:03d}/{max_epochs} "
            f"train_loss={train_step['loss']:.5f} train_MAE={train_metrics['mae']:.4f} "
            f"train_R2={train_metrics['r2']:.4f} val_MAE={val_metrics['mae']:.4f} "
            f"val_RMSE={val_metrics['rmse']:.4f} val_R2={val_metrics['r2']:.4f} "
            f"val_r={val_metrics['pearson_r']:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"throughput={train_step['throughput']:.1f}/s mem={train_step['peak_memory_gb']:.2f}GiB "
            f"patience={patience}/{patience_limit}{marker}", flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] stage={stage}", flush=True); break
    del execution_model
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint in {stage}")
    return best_state, best_mae


def train_seed(records, frame_index, seed, device_id, run_dir):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed); torch.cuda.set_device(device_id); torch.set_num_threads(4)
    device = torch.device(f"cuda:{device_id}")
    split_records = {
        name: records.loc[records.split.eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    train_augmented, train_augmented_loader = _loader(frame_index, split_records["train"], True)
    datasets, loaders = {}, {"train_augmented": train_augmented_loader}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(frame_index, split_records[split], False)
    model, head, weight_path = build_model()
    model = model.to(device, memory_format=torch.channels_last)
    total_parameters, _ = parameter_counts(model)
    print(
        f"[job-start] seed={seed} device={device} videos="
        f"{len(split_records['train'])}/{len(split_records['val'])}/{len(split_records['test'])} "
        f"patients={split_records['train'].hospital_id.nunique()}/"
        f"{split_records['val'].hospital_id.nunique()}/{split_records['test'].hospital_id.nunique()} "
        f"frames={len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} "
        f"train_inputs={train_augmented.model_input_count} views={len(TRAIN_VIEWS)} "
        f"parameters={total_parameters} patient_weighted_loss=true", flush=True,
    )
    history = []
    freeze_backbone(model)
    _, trainable = parameter_counts(model)
    print(f"[stage-start] stage=head lr={HEAD_LEARNING_RATE:.1e} trainable={trainable}", flush=True)
    head_optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    head_state, head_mae = _run_stage(
        "head", model, head, datasets, loaders, device, run_dir, history,
        HEAD_MAX_EPOCHS, HEAD_PATIENCE, head_optimizer,
    )
    model.load_state_dict(head_state)
    unfreeze_last_stage(model)
    _, trainable = parameter_counts(model)
    backbone_parameters = list(model.features[-2].parameters()) + list(model.features[-1].parameters())
    print(
        f"[stage-start] stage=last_stage head_lr={FINETUNE_HEAD_LEARNING_RATE:.1e} "
        f"backbone_lr={FINETUNE_BACKBONE_LEARNING_RATE:.1e} trainable={trainable}", flush=True,
    )
    finetune_optimizer = AdamW([
        {"params": backbone_parameters, "lr": FINETUNE_BACKBONE_LEARNING_RATE},
        {"params": model.classifier.parameters(), "lr": FINETUNE_HEAD_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    finetune_state, finetune_mae = _run_stage(
        "last_stage", model, head, datasets, loaders, device, run_dir, history,
        FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, finetune_optimizer,
    )
    selected_stage, selected_state = (
        ("last_stage", finetune_state) if finetune_mae <= head_mae else ("head", head_state)
    )
    model.load_state_dict(selected_state)
    metric_rows, predictions = [], []
    for split in ("train", "val", "test"):
        metrics, pred = _video_evaluation(model, datasets[split], loaders[split], device, split)
        metric_rows.append({"seed": seed, "selected_stage": selected_stage, "split": split, **metrics})
        pred.insert(0, "seed", seed); predictions.append(pred)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "video_predictions.csv", index=False)
    checkpoint = {
        "schema_version": 1, "seed": seed, "architecture": "efficientnet_b0",
        "selected_stage": selected_stage, "state_dict": selected_state,
        "pretrained_weight_path": str(weight_path),
    }
    torch.save(checkpoint, run_dir / "model.pt")
    _plot_history(history, run_dir / "training_history.png", seed)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "seed": seed, "device": str(device), "selected_stage": selected_stage,
        "head_best_val_mae": head_mae, "last_stage_best_val_mae": finetune_mae,
        "train_views": list(TRAIN_VIEWS), "frames_per_video": 20,
        "loss": "patient-weighted SmoothL1 on frame views",
        "evaluation": "mean prediction across 20 original frames per video",
    }, indent=2), encoding="utf-8")
    test = metric_rows[-1]
    print(
        f"[job-complete] seed={seed} selected={selected_stage} "
        f"test_MAE={test['mae']:.4f} test_RMSE={test['rmse']:.4f} "
        f"test_R2={test['r2']:.4f} test_r={test['pearson_r']:.4f}", flush=True,
    )
    return metric_rows
