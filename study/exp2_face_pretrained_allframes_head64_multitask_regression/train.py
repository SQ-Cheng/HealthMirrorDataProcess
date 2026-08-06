"""Two-stage training and masked video-level evaluation for one shared model."""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from study.exp2_face_pretrained_head32_regression.train import (
    _prepare_images,
    _regression_metrics,
)

from .config import (
    EVAL_BATCH_SIZES,
    EVAL_NUM_WORKERS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    GRAD_CLIP_NORM,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    MIN_LEARNING_RATE,
    PREFETCH_FACTOR,
    SMOOTH_L1_BETA,
    TARGETS,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZES,
    VIEW_NAMES,
    WEIGHT_DECAY,
)
from .data import GroupedFrameViewSampler, MultiTaskAllFramesDataset
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


def _clone_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _execution_model(model, architecture, stage, device):
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
            f"[compile-fallback] arch={architecture} stage={stage} "
            f"reason={type(exc).__name__}: {exc}"
        )
        return model, "eager"
    _log(
        f"[compile-enabled] arch={architecture} stage={stage} "
        f"mode={TORCH_COMPILE_MODE}"
    )
    return compiled, f"torch_compile:{TORCH_COMPILE_MODE}"


def _loader(frame_index, records, views, architecture, shuffle):
    interpolation = "bicubic" if architecture == "efficientnet_b0" else "bilinear"
    expand_all_views = shuffle and len(views) > 1
    batch_size = (
        TRAIN_SOURCE_BATCH_SIZES[architecture]
        if expand_all_views
        else EVAL_BATCH_SIZES[architecture]
    )
    num_workers = TRAIN_NUM_WORKERS if expand_all_views else EVAL_NUM_WORKERS
    dataset = MultiTaskAllFramesDataset(
        frame_index,
        records,
        views=views,
        interpolation=interpolation,
        expand_all_views=expand_all_views,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=GroupedFrameViewSampler(dataset) if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    )
    return dataset, loader


def _masked_loss(criterion, predictions, targets, masks, task_weights):
    weighted_mask = masks * task_weights.unsqueeze(0)
    numerator = (criterion(predictions, targets) * weighted_mask).sum()
    denominator = weighted_mask.sum()
    if denominator <= 0:
        raise RuntimeError("A batch contains no observed targets")
    return numerator / denominator, numerator.detach(), denominator.detach()


def _train_epoch(
    model,
    head,
    loader,
    optimizer,
    scaler,
    criterion,
    task_weights,
    device,
    encoder_frozen,
    max_batches=None,
):
    if encoder_frozen:
        model.eval()
        head.train()
    else:
        model.train()
    loss_numerator = 0.0
    loss_denominator = 0.0
    optimizer_steps = 0
    model_inputs = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch_index, (images, targets, masks, _, view_codes) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = _prepare_images(
            images, view_codes, loader.dataset.interpolation, device
        )
        if view_codes.ndim == 2:
            repeat = view_codes.shape[1]
            targets = targets.repeat_interleave(repeat, dim=0)
            masks = masks.repeat_interleave(repeat, dim=0)
        targets = targets.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            predictions = model(images)
            loss, numerator, denominator = _masked_loss(
                criterion, predictions, targets, masks, task_weights
            )
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
        loss_numerator += float(numerator.cpu())
        loss_denominator += float(denominator.cpu())
        model_inputs += len(targets)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    return (
        loss_numerator / max(loss_denominator, 1e-12),
        optimizer_steps,
        model_inputs,
        elapsed,
        peak_memory_gb,
    )


@torch.no_grad()
def _evaluate(model, loader, criterion, task_weights, device, max_batches=None):
    model.eval()
    loss_numerator = 0.0
    loss_denominator = 0.0
    predictions_out, targets_out, masks_out, record_indices = [], [], [], []
    for batch_index, (images, targets, masks, indices, view_codes) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = _prepare_images(
            images, view_codes, loader.dataset.interpolation, device
        )
        targets_device = targets.to(device, non_blocking=True)
        masks_device = masks.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            predictions = model(images)
            _, numerator, denominator = _masked_loss(
                criterion,
                predictions,
                targets_device,
                masks_device,
                task_weights,
            )
        loss_numerator += float(numerator.cpu())
        loss_denominator += float(denominator.cpu())
        predictions_out.append(predictions.float().cpu().numpy())
        targets_out.append(targets.numpy().copy())
        masks_out.append(masks.numpy().copy())
        record_indices.append(indices.numpy().copy())
    empty = np.empty((0, len(TARGETS)), dtype=np.float32)
    return {
        "loss": loss_numerator / max(loss_denominator, 1e-12),
        "predictions": (
            np.concatenate(predictions_out, axis=0) if predictions_out else empty
        ),
        "targets": np.concatenate(targets_out, axis=0) if targets_out else empty,
        "masks": np.concatenate(masks_out, axis=0) if masks_out else empty,
        "record_indices": (
            np.concatenate(record_indices)
            if record_indices
            else np.asarray([], dtype=np.int64)
        ),
    }


def _video_results(evaluation, dataset, split):
    sample_indices = evaluation["record_indices"].astype(np.int64)
    video_rows = dataset.frame_video_rows[sample_indices]
    unique_rows, inverse = np.unique(video_rows, return_inverse=True)
    sums = np.zeros((len(unique_rows), len(TARGETS)), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique_rows)).astype(np.int64)
    np.add.at(sums, inverse, evaluation["predictions"])
    mean_predictions = sums / counts[:, None]

    metric_rows = []
    prediction_frames = []
    for task_index, target in enumerate(TARGETS):
        observed = dataset.masks_by_video[unique_rows, task_index].astype(bool)
        selected_rows = unique_rows[observed]
        y_true = dataset.targets_by_video[selected_rows, task_index]
        y_pred = mean_predictions[observed, task_index]
        metrics = _regression_metrics(y_true, y_pred)
        metric_rows.append({"target": target, **metrics})
        metadata_columns = [
            "hospital_id",
            "video_id",
            f"{target}__binary_label",
            f"{target}__raw_value",
            f"{target}__score_threshold",
            f"{target}__score_scale",
        ]
        predictions = dataset.video_records.iloc[selected_rows][
            metadata_columns
        ].reset_index(drop=True)
        predictions = predictions.rename(
            columns={
                f"{target}__binary_label": "binary_label",
                f"{target}__raw_value": "raw_value",
                f"{target}__score_threshold": "score_threshold",
                f"{target}__score_scale": "score_scale",
            }
        )
        predictions.insert(0, "split", split)
        predictions.insert(1, "target", target)
        predictions["y_true"] = y_true
        predictions["y_pred"] = y_pred
        predictions["residual"] = y_pred - y_true
        predictions["frame_count"] = counts[observed]
        prediction_frames.append(predictions)

    numeric_keys = tuple(
        key
        for key in metric_rows[0]
        if key != "target"
    )
    macro = {"target": "macro"}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float64)
        if key in {"n", "sign_n", "tn", "fp", "fn", "tp"}:
            macro[key] = int(np.nansum(values))
        else:
            macro[key] = (
                float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            )
    return (
        metric_rows,
        macro,
        pd.concat(prediction_frames, ignore_index=True),
        {
            "video_rows": video_rows.astype(np.int32),
            "source_frame_index": dataset.index.source_indices[
                dataset.frame_indices[sample_indices]
            ].astype(np.int32),
            "targets": evaluation["targets"].astype(np.float32),
            "predictions": evaluation["predictions"].astype(np.float32),
            "masks": evaluation["masks"].astype(np.uint8),
        },
    )


def _plot_history(history, path, architecture):
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for stage, group in frame.groupby("stage", sort=False):
        axes[0].plot(group.global_epoch, group.train_loss, label=f"{stage} train")
        axes[0].plot(
            group.global_epoch,
            group.val_loss,
            linestyle="--",
            label=f"{stage} val",
        )
        axes[1].plot(
            group.global_epoch, group.train_macro_mae, label=f"{stage} train"
        )
        axes[1].plot(
            group.global_epoch,
            group.val_macro_mae,
            linestyle="--",
            label=f"{stage} val",
        )
        axes[2].plot(
            group.global_epoch, group.train_macro_pearson_r, label=f"{stage} train"
        )
        axes[2].plot(
            group.global_epoch,
            group.val_macro_pearson_r,
            linestyle="--",
            label=f"{stage} val",
        )
    axes[0].set_title("Task-balanced masked SmoothL1")
    axes[1].set_title("Macro video-level MAE")
    axes[2].set_title("Macro video-level Pearson r")
    axes[2].set_ylim(-1.05, 1.05)
    for axis in axes:
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"{architecture} / five-output regression")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _metric_history_columns(prefix, task_rows, macro):
    values = {
        f"{prefix}_macro_mae": macro["mae"],
        f"{prefix}_macro_rmse": macro["rmse"],
        f"{prefix}_macro_pearson_r": macro["pearson_r"],
        f"{prefix}_macro_spearman_r": macro["spearman_r"],
        f"{prefix}_macro_sign_bacc": macro["sign_balanced_accuracy"],
        f"{prefix}_macro_sign_auc": macro["sign_roc_auc"],
    }
    for row in task_rows:
        target = row["target"]
        values[f"{prefix}_{target}_mae"] = row["mae"]
        values[f"{prefix}_{target}_rmse"] = row["rmse"]
        values[f"{prefix}_{target}_pearson_r"] = row["pearson_r"]
        values[f"{prefix}_{target}_sign_bacc"] = row[
            "sign_balanced_accuracy"
        ]
        values[f"{prefix}_{target}_sign_auc"] = row["sign_roc_auc"]
    return values


def _run_stage(
    stage,
    model,
    head,
    loaders,
    datasets,
    criterion,
    task_weights,
    device,
    learning_rate,
    max_epochs,
    patience_limit,
    history,
    run_dir,
    encoder_frozen,
    architecture,
    max_batches,
):
    execution_model, execution_backend = _execution_model(
        model, architecture, stage, device
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
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
            task_weights,
            device,
            encoder_frozen,
            max_batches,
        )
        train_eval = _evaluate(
            execution_model,
            loaders["train"],
            criterion,
            task_weights,
            device,
            max_batches,
        )
        val_eval = _evaluate(
            execution_model,
            loaders["val"],
            criterion,
            task_weights,
            device,
            max_batches,
        )
        train_rows, train_macro, _, _ = _video_results(
            train_eval, datasets["train"], "train"
        )
        val_rows, val_macro, _, _ = _video_results(
            val_eval, datasets["val"], "val"
        )
        score = (
            -val_macro["mae"]
            if np.isfinite(val_macro["mae"])
            else -val_eval["loss"]
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "architecture": architecture,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "global_epoch": start_global_epoch + stage_epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_eval["loss"],
            "val_loss": val_eval["loss"],
            **_metric_history_columns("train", train_rows, train_macro),
            **_metric_history_columns("val", val_rows, val_macro),
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
            best_score = score
            best_state = _clone_state(model)
            patience = 0
            marker = "*"
        else:
            patience += 1
            marker = ""
        _log(
            f"[epoch] arch={architecture} stage={stage} "
            f"{stage_epoch:03d}/{max_epochs}: train_loss={train_loss:.4f} "
            f"train_macro_MAE={train_macro['mae']:.4f} "
            f"val_loss={val_eval['loss']:.4f} "
            f"val_macro_MAE={val_macro['mae']:.4f} "
            f"val_macro_r={val_macro['pearson_r']:.4f} "
            f"lr={current_lr:.2e} steps={optimizer_steps} "
            f"throughput={train_inputs / max(train_seconds, 1e-9):.1f}/s "
            f"peak_mem={peak_gpu_memory_gb:.2f}GiB "
            f"patience={patience}/{patience_limit}{marker}"
        )
        if optimizer_steps:
            scheduler.step()
        if patience >= patience_limit:
            _log(f"[early-stop] arch={architecture} stage={stage}")
            break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint for {architecture}/{stage}")
    del execution_model
    return best_state, best_score


def train_architecture(
    architecture,
    frame_index,
    records,
    task_weights,
    task_frame_counts,
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
        split: records.loc[records["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
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

    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    task_weights_tensor = torch.as_tensor(
        task_weights, dtype=torch.float32, device=device
    )
    model, head, weight_path = build_pretrained_model(architecture, weights_dir)
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA, reduction="none")
    total_parameters, _ = parameter_counts(model)
    history = []
    source_batch_size = TRAIN_SOURCE_BATCH_SIZES[architecture]
    _log(
        f"[job-start] arch={architecture} device={device} "
        f"train/val/test videos={len(records_by_split['train'])}/"
        f"{len(records_by_split['val'])}/{len(records_by_split['test'])} "
        f"source_frames={datasets['train'].frame_count}/"
        f"{datasets['val'].frame_count}/{datasets['test'].frame_count} "
        f"train_inputs={train_augmented_dataset.model_input_count} "
        f"source_batch={source_batch_size} effective_batch="
        f"{source_batch_size * len(VIEW_NAMES)} views={len(VIEW_NAMES)} "
        f"task_frame_counts={dict(zip(TARGETS, task_frame_counts))} "
        f"task_weights={dict(zip(TARGETS, np.round(task_weights, 6)))} "
        f"parameters={total_parameters}"
    )

    freeze_encoder(model, head)
    _, head_trainable = parameter_counts(model)
    _log(
        f"[stage-start] arch={architecture} stage=head "
        f"lr={HEAD_LEARNING_RATE:.1e} trainable={head_trainable}"
    )
    head_state, head_score = _run_stage(
        "head",
        model,
        head,
        loaders,
        datasets,
        criterion,
        task_weights_tensor,
        device,
        HEAD_LEARNING_RATE,
        head_epochs,
        head_patience,
        history,
        run_dir,
        True,
        architecture,
        max_batches,
    )
    model.load_state_dict(head_state)
    torch.save(
        {
            "model_state_dict": head_state,
            "architecture": architecture,
            "targets": TARGETS,
        },
        os.path.join(run_dir, "stage_head_best.pt"),
    )

    unfreeze_all(model)
    _, full_trainable = parameter_counts(model)
    _log(
        f"[stage-start] arch={architecture} stage=finetune "
        f"lr={FINETUNE_LEARNING_RATE:.1e} trainable={full_trainable}"
    )
    finetune_state, finetune_score = _run_stage(
        "finetune",
        model,
        head,
        loaders,
        datasets,
        criterion,
        task_weights_tensor,
        device,
        FINETUNE_LEARNING_RATE,
        finetune_epochs,
        finetune_patience,
        history,
        run_dir,
        False,
        architecture,
        max_batches,
    )
    torch.save(
        {
            "model_state_dict": finetune_state,
            "architecture": architecture,
            "targets": TARGETS,
        },
        os.path.join(run_dir, "stage_finetune_best.pt"),
    )
    selected_stage = "finetune" if finetune_score >= head_score else "head"
    selected_state = finetune_state if selected_stage == "finetune" else head_state
    model.load_state_dict(selected_state)

    evaluations = {
        split: _evaluate(
            model,
            loaders[split],
            criterion,
            task_weights_tensor,
            device,
            max_batches,
        )
        for split in ("train", "val", "test")
    }
    metric_rows, video_predictions, compact = [], [], []
    video_ids = records["video_id"].astype(str).drop_duplicates().to_numpy()
    video_code = {video_id: index for index, video_id in enumerate(video_ids)}
    split_code = {"train": 0, "val": 1, "test": 2}
    for split in ("train", "val", "test"):
        task_rows, macro, predictions, frames = _video_results(
            evaluations[split], datasets[split], split
        )
        for row in (*task_rows, macro):
            metric_rows.append(
                {
                    "architecture": architecture,
                    "target": row["target"],
                    "split": split,
                    "selected_stage": selected_stage,
                    **{key: value for key, value in row.items() if key != "target"},
                }
            )
        predictions.insert(0, "architecture", architecture)
        video_predictions.append(predictions)
        local_video_ids = datasets[split].video_records["video_id"].astype(str).to_numpy()
        compact.append(
            {
                "split_code": np.full(
                    len(frames["predictions"]), split_code[split], dtype=np.uint8
                ),
                "video_code": np.asarray(
                    [
                        video_code[local_video_ids[row]]
                        for row in frames["video_rows"]
                    ],
                    dtype=np.int32,
                ),
                **{key: value for key, value in frames.items() if key != "video_rows"},
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    pd.concat(video_predictions, ignore_index=True).to_csv(
        os.path.join(run_dir, "video_predictions.csv"), index=False
    )
    np.savez_compressed(
        os.path.join(run_dir, "frame_predictions.npz"),
        task_names=np.asarray(TARGETS, dtype=str),
        split_names=np.asarray(("train", "val", "test"), dtype=str),
        video_ids=video_ids,
        split_code=np.concatenate([item["split_code"] for item in compact]),
        video_code=np.concatenate([item["video_code"] for item in compact]),
        source_frame_index=np.concatenate(
            [item["source_frame_index"] for item in compact]
        ),
        y_true=np.concatenate([item["targets"] for item in compact]),
        y_pred=np.concatenate([item["predictions"] for item in compact]),
        target_mask=np.concatenate([item["masks"] for item in compact]),
        score_boundary=np.asarray([0.0], dtype=np.float32),
    )
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    _plot_history(history, os.path.join(run_dir, "history.png"), architecture)
    torch.save(
        {
            "model_state_dict": selected_state,
            "architecture": architecture,
            "targets": TARGETS,
            "selected_stage": selected_stage,
            "task_type": "multi_output_abnormal_score_regression",
            "task_frame_counts": dict(zip(TARGETS, map(int, task_frame_counts))),
            "task_loss_weights": dict(zip(TARGETS, map(float, task_weights))),
            "missing_target_policy": "masked; unavailable targets do not enter loss",
            "score_boundary": 0.0,
            "score_transform": "asinh",
            "smooth_l1_beta": SMOOTH_L1_BETA,
            "input_size": [224, 224],
            "normalization": "ImageNet mean/std",
            "frame_policy": "all decodable MJPEG frames streamed by byte offset",
            "training_views_per_frame": len(VIEW_NAMES),
            "pretrained_weight_file": os.path.basename(weight_path),
            "pretrained_weight_sha256": _sha256(weight_path),
        },
        os.path.join(run_dir, "model.pt"),
    )
    test_macro = metrics_frame.loc[
        metrics_frame["split"].eq("test") & metrics_frame["target"].eq("macro")
    ].iloc[0]
    _log(
        f"[job-done] arch={architecture} selected={selected_stage} "
        f"test_macro_MAE={test_macro['mae']:.4f} "
        f"test_macro_RMSE={test_macro['rmse']:.4f} "
        f"test_macro_r={test_macro['pearson_r']:.4f} "
        f"elapsed_min={(time.time() - start_time) / 60.0:.1f}"
    )
    del model
    for dataset in (train_augmented_dataset, *datasets.values()):
        dataset.close()
    del loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_frame
