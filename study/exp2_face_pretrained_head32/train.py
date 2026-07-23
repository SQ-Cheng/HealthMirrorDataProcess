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
    BATCH_SIZE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    GRAD_CLIP_NORM,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    MIN_LEARNING_RATE,
    NUM_WORKERS,
    POS_WEIGHT_MAX,
    POS_WEIGHT_MIN,
    VIEW_NAMES,
    WEIGHT_DECAY,
)
from .data import PretrainedFrameDataset
from .models import (
    build_pretrained_model,
    freeze_encoder,
    parameter_counts,
    unfreeze_all,
)


def _log(message):
    print(message, flush=True)


@lru_cache(maxsize=None)
def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(face, records, views, architecture, shuffle):
    interpolation = "bicubic" if architecture == "efficientnet_b0" else "bilinear"
    return DataLoader(
        PretrainedFrameDataset(face, records, views=views, interpolation=interpolation),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


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


def _train_epoch(
    model,
    head,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    encoder_frozen,
    max_batches=None,
):
    if encoder_frozen:
        model.eval()
        head.train()
    else:
        model.train()
    total_loss, batches, optimizer_steps = 0.0, 0, 0
    for batch_index, (images, labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
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
    return total_loss / max(batches, 1), optimizer_steps


@torch.no_grad()
def _evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    losses, scores, labels_out, record_indices = [], [], [], []
    for batch_index, (images, labels, indices) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
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


def _prediction_frames(evaluation, records, split):
    rows = records.iloc[evaluation["record_indices"]].copy().reset_index(drop=True)
    rows["split"] = split
    rows["score"] = evaluation["scores"]
    rows["y_true"] = evaluation["labels"].astype(int)
    return rows


def _aggregate_video_predictions(frame_predictions):
    return frame_predictions.groupby(["split", "video_id"], as_index=False).agg(
        hospital_id=("hospital_id", "first"),
        y_true=("y_true", "first"),
        score=("score", "mean"),
        frame_count=("frame_index", "count"),
    )


def _video_metrics(evaluation, records, split, threshold=0.5):
    frame_predictions = _prediction_frames(evaluation, records, split)
    video_predictions = _aggregate_video_predictions(frame_predictions)
    metrics = _binary_metrics(
        video_predictions["y_true"], video_predictions["score"], threshold
    )
    return metrics, frame_predictions, video_predictions


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
    records_by_split,
    criterion,
    device,
    learning_rate,
    max_epochs,
    patience_limit,
    history,
    run_dir,
    encoder_frozen,
    architecture,
    target,
    max_batches,
):
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
        train_loss, optimizer_steps = _train_epoch(
            model,
            head,
            loaders["train_augmented"],
            optimizer,
            scaler,
            criterion,
            device,
            encoder_frozen,
            max_batches=max_batches,
        )
        train_eval = _evaluate(
            model, loaders["train"], criterion, device, max_batches=max_batches
        )
        validation = _evaluate(
            model, loaders["val"], criterion, device, max_batches=max_batches
        )
        train_metrics, _, _ = _video_metrics(
            train_eval, records_by_split["train"], "train"
        )
        val_metrics, _, _ = _video_metrics(
            validation, records_by_split["val"], "val"
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
            f"steps={optimizer_steps} patience={patience}/{patience_limit}{marker}"
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
    return best_state, best_score


def train_task(
    architecture,
    target,
    face,
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
    loaders = {
        "train_augmented": _loader(
            face, records_by_split["train"], VIEW_NAMES, architecture, True
        ),
        "train": _loader(face, records_by_split["train"], ("original",), architecture, False),
        "val": _loader(face, records_by_split["val"], ("original",), architecture, False),
        "test": _loader(face, records_by_split["test"], ("original",), architecture, False),
    }
    train_labels = records_by_split["train"]["label"].to_numpy(np.float32)
    n_pos, n_neg = int((train_labels > 0.5).sum()), int((train_labels < 0.5).sum())
    pos_weight = float(np.clip(n_neg / max(n_pos, 1), POS_WEIGHT_MIN, POS_WEIGHT_MAX))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, head, weight_path = build_pretrained_model(architecture, weights_dir)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    total_parameters, _ = parameter_counts(model)
    history = []
    _log(
        f"[job-start] arch={architecture} task={target} device={device} "
        f"train/val/test frames={len(records_by_split['train'])}/"
        f"{len(records_by_split['val'])}/{len(records_by_split['test'])} "
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
        records_by_split,
        criterion,
        device,
        HEAD_LEARNING_RATE,
        head_epochs,
        head_patience,
        history,
        run_dir,
        True,
        architecture,
        target,
        max_batches,
    )
    model.load_state_dict(head_state)
    torch.save(
        {"model_state_dict": head_state, "architecture": architecture, "target": target},
        os.path.join(run_dir, "stage_head_best.pt"),
    )

    unfreeze_all(model)
    _, full_trainable = parameter_counts(model)
    _log(
        f"[stage-start] arch={architecture} task={target} stage=finetune "
        f"lr={FINETUNE_LEARNING_RATE:.1e} trainable={full_trainable}"
    )
    finetune_state, finetune_score = _run_stage(
        "finetune",
        model,
        head,
        loaders,
        records_by_split,
        criterion,
        device,
        FINETUNE_LEARNING_RATE,
        finetune_epochs,
        finetune_patience,
        history,
        run_dir,
        False,
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
        evaluations["val"], records_by_split["val"], "val"
    )
    threshold = _optimal_threshold(validation_videos["y_true"], validation_videos["score"])
    metric_rows, frame_predictions, video_predictions = [], [], []
    for split in ("train", "val", "test"):
        metrics, frames, videos = _video_metrics(
            evaluations[split], records_by_split[split], split, threshold
        )
        metric_rows.append({
            "architecture": architecture,
            "target": target,
            "split": split,
            "selected_stage": selected_stage,
            **metrics,
        })
        frames["architecture"], frames["target"] = architecture, target
        videos["architecture"], videos["target"] = architecture, target
        frames["threshold"] = threshold
        videos["threshold"] = threshold
        frame_predictions.append(frames)
        video_predictions.append(videos)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.concat(frame_predictions, ignore_index=True).to_csv(
        os.path.join(run_dir, "frame_predictions.csv"), index=False
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_frame
