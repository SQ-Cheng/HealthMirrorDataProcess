"""Two-stage training and video-level evaluation for one backbone/target pair."""

from functools import lru_cache
import hashlib
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import (
    BRIGHTNESS_DELTA,
    CONTRAST_DELTA,
    CROP_SCALE,
    EVAL_BATCH_SIZES,
    FINETUNE_LEARNING_RATE,
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
    EVAL_NUM_WORKERS,
    PREFETCH_FACTOR,
    POS_WEIGHT_MAX,
    POS_WEIGHT_MIN,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZES,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    VIEW_NAMES,
    WEIGHT_DECAY,
)
from .data import AllFramesDataset, GroupedFrameViewSampler
from .models import (
    build_pretrained_model,
    configure_training_mode,
    freeze_encoder,
    parameter_counts,
    unfreeze_last_stage,
)


def _log(message):
    print(message, flush=True)


def _execution_model(model, architecture, target, stage, device):
    if device.type != "cuda" or not TORCH_COMPILE_ENABLED:
        return model, "eager"
    try:
        torch._dynamo.config.suppress_errors = True
        compiled = torch.compile(
            model,
            mode=TORCH_COMPILE_MODE,
            fullgraph=False,
            dynamic=False,
        )
    except Exception as exc:
        _log(
            f"[compile-fallback] arch={architecture} task={target} stage={stage} "
            f"reason={type(exc).__name__}: {exc}"
        )
        return model, "eager"
    _log(
        f"[compile-enabled] arch={architecture} task={target} stage={stage} "
        f"mode={TORCH_COMPILE_MODE}"
    )
    return compiled, f"torch_compile:{TORCH_COMPILE_MODE}"


@lru_cache(maxsize=None)
def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(frame_index, records, views, architecture, shuffle):
    interpolation = "bicubic" if architecture == "efficientnet_b0" else "bilinear"
    expand_all_views = shuffle and len(views) > 1
    batch_size = (
        TRAIN_SOURCE_BATCH_SIZES[architecture]
        if expand_all_views
        else EVAL_BATCH_SIZES[architecture]
    )
    num_workers = TRAIN_NUM_WORKERS if expand_all_views else EVAL_NUM_WORKERS
    dataset = AllFramesDataset(
        frame_index,
        records,
        views=views,
        interpolation=interpolation,
        expand_all_views=expand_all_views,
    )
    sampler = GroupedFrameViewSampler(dataset) if shuffle else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    )
    return dataset, loader


def _binary_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {
            "n": int(len(labels)),
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "threshold": float(threshold),
        }
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def _optimal_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    finite_indices = np.flatnonzero(finite)
    best = finite_indices[np.argmax((true_positive_rate - false_positive_rate)[finite])]
    return float(thresholds[best])


def _prepare_images(images, view_codes, interpolation, device):
    """Apply deterministic views, resize, and normalize on the assigned GPU."""
    images = images.to(device, non_blocking=True).float().div_(255.0)
    view_codes = view_codes.to(device, non_blocking=True)
    if view_codes.ndim == 2:
        images = images.repeat_interleave(view_codes.shape[1], dim=0)
        view_codes = view_codes.flatten()
    flip_mask = view_codes.eq(1)
    if flip_mask.any():
        images[flip_mask] = torch.flip(images[flip_mask], dims=(-1,))
    brightness_mask = view_codes.eq(3)
    if brightness_mask.any():
        images[brightness_mask] = (
            images[brightness_mask] * (1.0 + BRIGHTNESS_DELTA)
        ).clamp_(0.0, 1.0)
    contrast_mask = view_codes.eq(4)
    if contrast_mask.any():
        selected = images[contrast_mask]
        spatial_mean = selected.mean(dim=(-2, -1), keepdim=True)
        images[contrast_mask] = (
            (selected - spatial_mean) * (1.0 + CONTRAST_DELTA) + spatial_mean
        ).clamp_(0.0, 1.0)

    crop_mask = view_codes.eq(2)
    output = torch.empty(
        (len(images), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=images.dtype, device=device
    )
    full_mask = ~crop_mask
    if full_mask.any():
        output[full_mask] = functional.interpolate(
            images[full_mask],
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode=interpolation,
            align_corners=False,
            antialias=True,
        )
    if crop_mask.any():
        height, width = images.shape[-2:]
        crop_h = max(8, int(round(height * CROP_SCALE)))
        crop_w = max(8, int(round(width * CROP_SCALE)))
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        output[crop_mask] = functional.interpolate(
            images[crop_mask, :, top:top + crop_h, left:left + crop_w],
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode=interpolation,
            align_corners=False,
            antialias=True,
        )
    mean = output.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = output.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((output - mean) / std).contiguous(memory_format=torch.channels_last)


def _train_epoch(
    model,
    head,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    training_scope,
    architecture,
    max_batches=None,
):
    configure_training_mode(
        model,
        head,
        architecture,
        training_scope,
    )
    total_loss, batches, optimizer_steps, model_inputs = 0.0, 0, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch_index, (images, labels, _, view_codes) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = _prepare_images(
            images, view_codes, loader.dataset.interpolation, device
        )
        if view_codes.ndim == 2:
            labels = labels.repeat_interleave(view_codes.shape[1])
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            optimizer_steps += 1
        total_loss += float(loss.detach().cpu())
        batches += 1
        model_inputs += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    return (
        total_loss / max(batches, 1),
        optimizer_steps,
        model_inputs,
        elapsed,
        peak_memory_gb,
    )


@torch.no_grad()
def _evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    losses, scores, labels_out, record_indices = [], [], [], []
    for batch_index, (images, labels, indices, view_codes) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = _prepare_images(
            images, view_codes, loader.dataset.interpolation, device
        )
        labels_device = labels.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, labels_device)
        losses.append(float(loss.cpu()))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        labels_out.append(labels.numpy().ravel())
        record_indices.append(indices.numpy().ravel())
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "scores": np.concatenate(scores) if scores else np.asarray([], dtype=np.float32),
        "labels": (
            np.concatenate(labels_out) if labels_out else np.asarray([], dtype=np.float32)
        ),
        "record_indices": (
            np.concatenate(record_indices)
            if record_indices
            else np.asarray([], dtype=np.int64)
        ),
    }


def _video_metrics(evaluation, dataset, split, threshold=0.5):
    sample_indices = evaluation["record_indices"].astype(np.int64)
    video_rows = dataset.frame_video_rows[sample_indices]
    aggregation = pd.DataFrame({
        "video_row": video_rows,
        "y_true": evaluation["labels"].astype(int),
        "score": evaluation["scores"],
    }).groupby("video_row", as_index=False).agg(
        y_true=("y_true", "first"),
        score=("score", "mean"),
        frame_count=("score", "size"),
    )
    video_predictions = dataset.video_records.iloc[
        aggregation["video_row"].to_numpy(np.int64)
    ][["hospital_id", "video_id"]].reset_index(drop=True)
    video_predictions.insert(0, "split", split)
    video_predictions["y_true"] = aggregation["y_true"].to_numpy(np.int64)
    video_predictions["score"] = aggregation["score"].to_numpy(np.float32)
    video_predictions["frame_count"] = aggregation["frame_count"].to_numpy(np.int64)
    metrics = _binary_metrics(
        video_predictions["y_true"], video_predictions["score"], threshold
    )
    compact_frames = {
        "split": np.full(len(sample_indices), split),
        "video_row": video_rows.astype(np.int32),
        "source_frame_index": dataset.index.source_indices[
            dataset.frame_indices[sample_indices]
        ].astype(np.int32),
        "y_true": evaluation["labels"].astype(np.uint8),
        "score": evaluation["scores"].astype(np.float32),
    }
    return metrics, compact_frames, video_predictions


def _clone_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _plot_history(history, path, architecture, target):
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for stage, group in frame.groupby("stage", sort=False):
        axes[0].plot(group.global_epoch, group.train_loss, label=f"{stage} train")
        axes[0].plot(group.global_epoch, group.val_loss, linestyle="--", label=f"{stage} val")
        axes[1].plot(group.global_epoch, group.train_bacc, label=f"{stage} train bACC")
        axes[1].plot(group.global_epoch, group.train_roc_auc, label=f"{stage} train AUC")
        axes[1].plot(group.global_epoch, group.val_bacc, linestyle="--", label=f"{stage} val bACC")
        axes[1].plot(group.global_epoch, group.val_roc_auc, linestyle="--", label=f"{stage} val AUC")
    axes[0].set_title("Loss")
    axes[1].set_title("Video-level scores")
    for axis in axes:
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[1].set_ylim(-0.05, 1.05)
    figure.suptitle(f"{architecture} / {target}")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _run_stage(
    stage,
    model,
    head,
    loaders,
    datasets,
    criterion,
    device,
    learning_rate,
    max_epochs,
    patience_limit,
    history,
    run_dir,
    training_scope,
    architecture,
    target,
    max_batches,
):
    execution_model, execution_backend = _execution_model(
        model, architecture, target, stage, device
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(max_epochs, 1), eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=1024.0
    )
    best_score, best_state, patience = -np.inf, None, 0
    start_global_epoch = len(history)
    for stage_epoch in range(1, max_epochs + 1):
        (
            train_loss,
            optimizer_steps,
            train_inputs,
            train_seconds,
            peak_gpu_memory_gb,
        ) = _train_epoch(
            execution_model,
            head,
            loaders["train_augmented"],
            optimizer,
            scaler,
            criterion,
            device,
            training_scope,
            architecture,
            max_batches=max_batches,
        )
        train_eval = _evaluate(
            execution_model,
            loaders["train"],
            criterion,
            device,
            max_batches=max_batches,
        )
        validation = _evaluate(
            execution_model,
            loaders["val"],
            criterion,
            device,
            max_batches=max_batches,
        )
        train_metrics, _, _ = _video_metrics(
            train_eval, datasets["train"], "train"
        )
        val_metrics, _, _ = _video_metrics(
            validation, datasets["val"], "val"
        )
        score = (
            val_metrics["roc_auc"]
            if np.isfinite(val_metrics["roc_auc"])
            else -validation["loss"]
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "architecture": architecture,
            "target": target,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "global_epoch": start_global_epoch + stage_epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval["loss"],
            "train_bacc": train_metrics["balanced_accuracy"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": validation["loss"],
            "val_bacc": val_metrics["balanced_accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "learning_rate": current_lr,
            "optimizer_steps": optimizer_steps,
            "train_model_inputs": train_inputs,
            "train_seconds": train_seconds,
            "train_inputs_per_second": train_inputs / max(train_seconds, 1e-9),
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "execution_backend": execution_backend,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        if score > best_score + 1e-4:
            best_score, best_state, patience, marker = score, _clone_state(model), 0, "*"
        else:
            patience, marker = patience + 1, ""
        _log(
            f"[epoch] arch={architecture} task={target} stage={stage} "
            f"{stage_epoch:03d}/{max_epochs}: train_loss={train_loss:.4f} "
            f"train_AUC={train_metrics['roc_auc']:.4f} "
            f"train_bACC={train_metrics['balanced_accuracy']:.4f} "
            f"val_loss={validation['loss']:.4f} val_AUC={val_metrics['roc_auc']:.4f} "
            f"val_bACC={val_metrics['balanced_accuracy']:.4f} lr={current_lr:.2e} "
            f"steps={optimizer_steps} "
            f"throughput={train_inputs / max(train_seconds, 1e-9):.1f}/s "
            f"peak_mem={peak_gpu_memory_gb:.2f}GiB "
            f"patience={patience}/{patience_limit}{marker}"
        )
        if optimizer_steps > 0:
            scheduler.step()
        else:
            _log(f"[no-update] arch={architecture} task={target} stage={stage} "
                 "all optimizer steps were skipped by GradScaler")
        if patience >= patience_limit:
            _log(f"[early-stop] arch={architecture} task={target} stage={stage}")
            break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint for {architecture}/{target}/{stage}")
    del execution_model
    return best_state, best_score


def train_task(
    architecture,
    target,
    frame_index,
    records,
    weights_dir,
    run_dir,
    head_epochs=HEAD_MAX_EPOCHS,
    finetune_epochs=FINETUNE_MAX_EPOCHS,
    head_patience=HEAD_PATIENCE,
    finetune_patience=FINETUNE_PATIENCE,
    max_batches=None,
):
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.time()
    records_by_split = {
        name: records.loc[records["split"].eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    train_augmented_dataset, train_augmented_loader = _loader(
        frame_index, records_by_split["train"], VIEW_NAMES, architecture, True
    )
    datasets, loaders = {}, {"train_augmented": train_augmented_loader}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(
            frame_index,
            records_by_split[split],
            ("original",),
            architecture,
            False,
        )
    train_labels = datasets["train"].frame_labels()
    n_pos, n_neg = int((train_labels > 0.5).sum()), int((train_labels < 0.5).sum())
    pos_weight = float(np.clip(n_neg / max(n_pos, 1), POS_WEIGHT_MIN, POS_WEIGHT_MAX))
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    model, head, weight_path = build_pretrained_model(architecture, weights_dir)
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    total_parameters, _ = parameter_counts(model)
    history = []
    source_batch_size = TRAIN_SOURCE_BATCH_SIZES[architecture]
    _log(
        f"[job-start] arch={architecture} task={target} device={device} "
        f"train/val/test videos={len(records_by_split['train'])}/"
        f"{len(records_by_split['val'])}/{len(records_by_split['test'])} "
        f"source_frames={datasets['train'].frame_count}/"
        f"{datasets['val'].frame_count}/{datasets['test'].frame_count} "
        f"train_inputs={train_augmented_dataset.model_input_count} "
        f"source_batch={source_batch_size} effective_batch="
        f"{source_batch_size * len(VIEW_NAMES)} views={len(VIEW_NAMES)} "
        f"pos_weight={pos_weight:.3f} parameters={total_parameters}"
    )

    freeze_encoder(model, head)
    _, head_trainable = parameter_counts(model)
    _log(
        f"[stage-start] arch={architecture} task={target} stage=head "
        f"lr={HEAD_LEARNING_RATE:.1e} trainable={head_trainable}"
    )
    head_state, head_score = _run_stage(
        "head",
        model,
        head,
        loaders,
        datasets,
        criterion,
        device,
        HEAD_LEARNING_RATE,
        head_epochs,
        head_patience,
        history,
        run_dir,
        "head",
        architecture,
        target,
        max_batches,
    )
    model.load_state_dict(head_state)
    torch.save(
        {"model_state_dict": head_state, "architecture": architecture, "target": target},
        os.path.join(run_dir, "stage_head_best.pt"),
    )

    unfreeze_last_stage(model, head, architecture)
    _, partial_trainable = parameter_counts(model)
    _log(
        f"[stage-start] arch={architecture} task={target} stage=finetune "
        f"scope=last_backbone_stage lr={FINETUNE_LEARNING_RATE:.1e} "
        f"trainable={partial_trainable}"
    )
    finetune_state, finetune_score = _run_stage(
        "finetune",
        model,
        head,
        loaders,
        datasets,
        criterion,
        device,
        FINETUNE_LEARNING_RATE,
        finetune_epochs,
        finetune_patience,
        history,
        run_dir,
        "last_stage",
        architecture,
        target,
        max_batches,
    )
    torch.save(
        {
            "model_state_dict": finetune_state,
            "architecture": architecture,
            "target": target,
        },
        os.path.join(run_dir, "stage_finetune_best.pt"),
    )
    selected_stage = "finetune" if finetune_score >= head_score else "head"
    selected_state = finetune_state if selected_stage == "finetune" else head_state
    model.load_state_dict(selected_state)

    evaluations = {
        name: _evaluate(model, loaders[name], criterion, device, max_batches=max_batches)
        for name in ("train", "val", "test")
    }
    _, _, validation_videos = _video_metrics(
        evaluations["val"], datasets["val"], "val"
    )
    threshold = _optimal_threshold(validation_videos["y_true"], validation_videos["score"])
    metric_rows, compact_predictions, video_predictions = [], [], []
    video_ids = records["video_id"].astype(str).drop_duplicates().to_numpy()
    video_code = {video_id: index for index, video_id in enumerate(video_ids)}
    split_code = {"train": 0, "val": 1, "test": 2}
    for split in ("train", "val", "test"):
        metrics, frames, videos = _video_metrics(
            evaluations[split], datasets[split], split, threshold
        )
        metric_rows.append({
            "architecture": architecture,
            "target": target,
            "split": split,
            "selected_stage": selected_stage,
            **metrics,
        })
        videos["architecture"], videos["target"] = architecture, target
        videos["threshold"] = threshold
        local_video_ids = datasets[split].video_records["video_id"].astype(str).to_numpy()
        compact_predictions.append({
            "split_code": np.full(len(frames["score"]), split_code[split], dtype=np.uint8),
            "video_code": np.asarray(
                [video_code[local_video_ids[index]] for index in frames["video_row"]],
                dtype=np.int32,
            ),
            "source_frame_index": frames["source_frame_index"],
            "y_true": frames["y_true"],
            "score": frames["score"],
        })
        video_predictions.append(videos)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    np.savez_compressed(
        os.path.join(run_dir, "frame_predictions.npz"),
        split_names=np.asarray(("train", "val", "test"), dtype=str),
        video_ids=video_ids,
        split_code=np.concatenate([item["split_code"] for item in compact_predictions]),
        video_code=np.concatenate([item["video_code"] for item in compact_predictions]),
        source_frame_index=np.concatenate([
            item["source_frame_index"] for item in compact_predictions
        ]),
        y_true=np.concatenate([item["y_true"] for item in compact_predictions]),
        score=np.concatenate([item["score"] for item in compact_predictions]),
        threshold=np.asarray([threshold], dtype=np.float32),
    )
    pd.concat(video_predictions, ignore_index=True).to_csv(
        os.path.join(run_dir, "video_predictions.csv"), index=False
    )
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    _plot_history(
        history, os.path.join(run_dir, "history.png"), architecture, target
    )
    torch.save({
        "model_state_dict": selected_state,
        "architecture": architecture,
        "target": target,
        "selected_stage": selected_stage,
        "threshold": threshold,
        "input_size": [224, 224],
        "normalization": "ImageNet mean/std",
        "frame_policy": "all decodable MJPEG frames streamed by byte offset",
        "training_views_per_frame": len(VIEW_NAMES),
        "pretrained_weight_file": os.path.basename(weight_path),
        "pretrained_weight_sha256": _sha256(weight_path),
    }, os.path.join(run_dir, "model.pt"))
    test_row = metrics_frame.loc[metrics_frame["split"].eq("test")].iloc[0]
    _log(
        f"[job-done] arch={architecture} task={target} selected={selected_stage} "
        f"test_AUC={test_row['roc_auc']:.4f} "
        f"test_bACC={test_row['balanced_accuracy']:.4f} "
        f"elapsed_min={(time.time() - start_time) / 60.0:.1f}"
    )
    del model
    for dataset in (train_augmented_dataset, *datasets.values()):
        dataset.close()
    del loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_frame
