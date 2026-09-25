"""Two-stage training and pair-level evaluation for Exp6."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import (
    BRIGHTNESS_DELTA,
    CONTRAST_DELTA,
    CROP_SCALE,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
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
    PREFETCH_FACTOR,
    SMOOTH_L1_BETA,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZE,
    VIEWS,
    WEIGHT_DECAY,
)
from .data import ChunkShuffleSampler, PairedFrameDataset
from .models import build_model, freeze_backbone, parameter_counts, unfreeze_all


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _loader(frame_index, records, train, train_views=VIEWS):
    dataset = PairedFrameDataset(
        frame_index,
        records,
        views=train_views if train else ("original",),
        expand_views=train,
    )
    workers = TRAIN_NUM_WORKERS if train else EVAL_NUM_WORKERS
    return dataset, DataLoader(
        dataset,
        batch_size=TRAIN_SOURCE_BATCH_SIZE if train else EVAL_BATCH_SIZE,
        sampler=ChunkShuffleSampler(dataset) if train else None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
    )


def _prepare(images, codes, device):
    images = images.to(device, non_blocking=True).float().div_(255.0)
    codes = codes.to(device, non_blocking=True)
    if codes.ndim == 2:
        images = images.repeat_interleave(codes.shape[1], dim=0)
        codes = codes.flatten()
    flip = codes.eq(1)
    if flip.any():
        images[flip] = torch.flip(images[flip], dims=(-1,))
    brightness = codes.eq(3)
    if brightness.any():
        images[brightness] = (
            images[brightness] * (1.0 + BRIGHTNESS_DELTA)
        ).clamp_(0.0, 1.0)
    contrast = codes.eq(4)
    if contrast.any():
        selected = images[contrast]
        mean = selected.mean(dim=(-2, -1), keepdim=True)
        images[contrast] = (
            (selected - mean) * (1.0 + CONTRAST_DELTA) + mean
        ).clamp_(0.0, 1.0)
    crop = codes.eq(2)
    output = torch.empty((len(images), 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    regular = ~crop
    if regular.any():
        output[regular] = F.interpolate(
            images[regular], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    if crop.any():
        height, width = images.shape[-2:]
        crop_h = max(8, round(height * CROP_SCALE))
        crop_w = max(8, round(width * CROP_SCALE))
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        output[crop] = F.interpolate(
            images[crop, :, top:top + crop_h, left:left + crop_w],
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    mean = output.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = output.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((output - mean) / std).contiguous(memory_format=torch.channels_last)


def _weighted_loss(prediction, target, weights):
    values = F.smooth_l1_loss(
        prediction, target, beta=SMOOTH_L1_BETA, reduction="none"
    )
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _compile(model, target, stage):
    if not TORCH_COMPILE_ENABLED:
        return model, "eager"
    try:
        torch._dynamo.config.suppress_errors = True
        compiled = torch.compile(
            model, mode=TORCH_COMPILE_MODE, fullgraph=False, dynamic=False
        )
        print(
            f"[compile-enabled] target={target} stage={stage} "
            f"mode={TORCH_COMPILE_MODE}", flush=True
        )
        return compiled, f"torch_compile:{TORCH_COMPILE_MODE}"
    except Exception as exc:
        print(
            f"[compile-fallback] target={target} stage={stage} "
            f"reason={type(exc).__name__}: {exc}", flush=True
        )
        return model, "eager"


def _train_epoch(model, raw_model, loader, optimizer, scaler, device, frozen, max_batches):
    if frozen:
        model.eval()
        raw_model.head.train()
    else:
        model.train()
    total = weight_total = inputs = batches = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for batch_index, (first, second, target, _, codes, weights) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        first = _prepare(first, codes, device)
        second = _prepare(second, codes, device)
        repeat = codes.shape[1]
        target = target.repeat_interleave(repeat).to(device, non_blocking=True)
        weights = weights.repeat_interleave(repeat).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(first, second).squeeze(1)
            loss = _weighted_loss(prediction, target, weights)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        batch_weight = float(weights.sum().detach().cpu())
        total += float(loss.detach().cpu()) * batch_weight
        weight_total += batch_weight
        inputs += len(target)
        batches += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss": total / max(weight_total, 1e-8),
        "batches": int(batches),
        "inputs": int(inputs),
        "seconds": elapsed,
        "inputs_per_second": inputs / max(elapsed, 1e-8),
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


@torch.no_grad()
def _frame_predictions(model, loader, device, max_batches=None):
    model.eval()
    truth, prediction, frame_rows = [], [], []
    for batch_index, (first, second, target, rows, codes, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        first = _prepare(first, codes, device)
        second = _prepare(second, codes, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(first, second).squeeze(1)
        truth.append(target.numpy())
        prediction.append(output.float().cpu().numpy())
        frame_rows.append(rows.numpy())
    return np.concatenate(truth), np.concatenate(prediction), np.concatenate(frame_rows)


def _metrics(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    result = {
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "explained_variance": (
            float(explained_variance_score(truth, prediction))
            if len(truth) > 1 else np.nan
        ),
        "pearson_r": (
            float(np.corrcoef(truth, prediction)[0, 1])
            if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0
            else np.nan
        ),
        "spearman_r": (
            float(pd.Series(truth).rank().corr(pd.Series(prediction).rank()))
            if len(truth) > 1 else np.nan
        ),
    }
    nonzero = ~np.isclose(truth, 0.0, atol=1e-12)
    labels = truth[nonzero] > 0
    decisions = prediction[nonzero]
    predicted_labels = decisions > 0
    result["direction_n"] = int(nonzero.sum())
    result["direction_accuracy"] = float((predicted_labels == labels).mean())
    result["direction_balanced_accuracy"] = (
        float(balanced_accuracy_score(labels, predicted_labels))
        if len(np.unique(labels)) == 2 else np.nan
    )
    result["direction_roc_auc"] = (
        float(roc_auc_score(labels, decisions))
        if len(np.unique(labels)) == 2 else np.nan
    )
    return result


def _evaluate(model, dataset, loader, device, split, scaler, max_batches=None):
    truth, prediction, rows = _frame_predictions(model, loader, device, max_batches)
    pair_rows = dataset.frame_pair_rows[rows]
    aggregate = pd.DataFrame({
        "pair_row": pair_rows,
        "y_true_scaled": truth,
        "y_pred_scaled": prediction,
    }).groupby("pair_row", as_index=False).agg(
        y_true_scaled=("y_true_scaled", "first"),
        y_pred_scaled=("y_pred_scaled", "mean"),
        frame_count=("y_pred_scaled", "size"),
        frame_prediction_std=("y_pred_scaled", "std"),
    )
    info = dataset.records.iloc[aggregate.pair_row.to_numpy(int)][[
        "pair_id", "target", "hospital_id", "first_video_id", "second_video_id",
        "first_value", "second_value", "raw_delta", "lab_interval_h",
        "first_match_delta_h", "second_match_delta_h",
    ]].reset_index(drop=True)
    aggregate["y_true"] = aggregate.y_true_scaled * scaler["iqr"] + scaler["median"]
    aggregate["y_pred"] = aggregate.y_pred_scaled * scaler["iqr"] + scaler["median"]
    result = pd.concat([info, aggregate.drop(columns="pair_row")], axis=1)
    result.insert(0, "split", split)
    return _metrics(result.y_true, result.y_pred), result


def _clone(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _stage(
    stage, raw_model, datasets, loaders, target, device, run_dir, scaler,
    history, epochs, patience_limit, optimizer, max_batches,
):
    execution, backend = _compile(raw_model, target, stage)
    amp_scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE)
    best_state, best_mae, patience, offset = None, np.inf, 0, len(history)
    for epoch in range(1, epochs + 1):
        step = _train_epoch(
            execution, raw_model, loaders["train_augmented"], optimizer,
            amp_scaler, device, stage == "head", max_batches,
        )
        train_metrics, _ = _evaluate(
            execution, datasets["train"], loaders["train"], device,
            "train", scaler, max_batches,
        )
        val_metrics, _ = _evaluate(
            execution, datasets["val"], loaders["val"], device,
            "val", scaler, max_batches,
        )
        row = {
            "target": target, "stage": stage, "stage_epoch": epoch,
            "global_epoch": offset + epoch,
            "train_optimization_loss": step["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_model_inputs": step["inputs"],
            "train_seconds": step["seconds"],
            "train_inputs_per_second": step["inputs_per_second"],
            "peak_gpu_memory_gb": step["peak_memory_gb"],
            "execution_backend": backend,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_metrics["mae"] < best_mae - 1e-8:
            best_mae, best_state, patience, marker = (
                val_metrics["mae"], _clone(raw_model), 0, "*"
            )
        else:
            patience += 1
            marker = ""
        print(
            f"[epoch] target={target} stage={stage} {epoch:03d}/{epochs} "
            f"train_MAE={train_metrics['mae']:.5g} val_MAE={val_metrics['mae']:.5g} "
            f"val_R2={val_metrics['r2']:.4f} val_r={val_metrics['pearson_r']:.4f} "
            f"val_dir_bACC={val_metrics['direction_balanced_accuracy']:.4f} "
            f"throughput={step['inputs_per_second']:.1f}/s "
            f"mem={step['peak_memory_gb']:.2f}GiB patience={patience}/{patience_limit}{marker}",
            flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] target={target} stage={stage}", flush=True)
            break
    del execution
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {target}/{stage}")
    return best_state, best_mae


def train_task(
    target, records, scaler, frame_index, device_id, run_dir, seed,
    head_epochs=HEAD_MAX_EPOCHS, finetune_epochs=FINETUNE_MAX_EPOCHS,
    max_batches=None, model_variant="shared", train_views=VIEWS,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.cuda.set_device(device_id)
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{device_id}")
    split_records = {
        name: records.loc[records.split.eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    train_augmented, train_augmented_loader = _loader(
        frame_index, split_records["train"], True, train_views
    )
    datasets, loaders = {}, {"train_augmented": train_augmented_loader}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(
            frame_index, split_records[split], False
        )
    model, weight_path = build_model(model_variant)
    model = model.to(device, memory_format=torch.channels_last)
    freeze_backbone(model)
    total, trainable = parameter_counts(model)
    print(
        f"[job-start] target={target} variant={model_variant} device={device} "
        f"pairs={len(split_records['train'])}/{len(split_records['val'])}/"
        f"{len(split_records['test'])} patients="
        f"{split_records['train'].hospital_id.nunique()}/"
        f"{split_records['val'].hospital_id.nunique()}/"
        f"{split_records['test'].hospital_id.nunique()} "
        f"frames={len(datasets['train'])}/{len(datasets['val'])}/"
        f"{len(datasets['test'])} train_inputs={train_augmented.model_input_count} "
        f"parameters={total} head_trainable={trainable}", flush=True,
    )
    history = []
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    print(
        f"[stage-start] target={target} stage=head lr={HEAD_LEARNING_RATE:.1e}",
        flush=True,
    )
    head_state, head_mae = _stage(
        "head", model, datasets, loaders, target, device, run_dir, scaler,
        history, head_epochs, HEAD_PATIENCE, optimizer, max_batches,
    )
    model.load_state_dict(head_state)
    unfreeze_all(model)
    _, trainable = parameter_counts(model)
    optimizer = AdamW(
        model.parameters(), lr=FINETUNE_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    print(
        f"[stage-start] target={target} stage=finetune "
        f"lr={FINETUNE_LEARNING_RATE:.1e} trainable={trainable}", flush=True,
    )
    fine_state, fine_mae = _stage(
        "finetune", model, datasets, loaders, target, device, run_dir, scaler,
        history, finetune_epochs, FINETUNE_PATIENCE, optimizer, max_batches,
    )
    selected_stage, selected_state = (
        ("finetune", fine_state) if fine_mae <= head_mae else ("head", head_state)
    )
    model.load_state_dict(selected_state)
    metric_rows, prediction_frames = [], []
    for split in ("train", "val", "test"):
        metrics, predictions = _evaluate(
            model, datasets[split], loaders[split], device, split, scaler,
            max_batches=None,
        )
        metric_rows.append({
            "target": target, "split": split,
            "selected_stage": selected_stage, **metrics,
        })
        prediction_frames.append(predictions)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        run_dir / "pair_predictions.csv", index=False
    )
    torch.save({
        "schema_version": 1,
        "experiment": "exp6_paired_face_lab_delta_regression",
        "architecture": (
            "independent_dual_efficientnet_b0_difference_head32"
            if model_variant == "independent_backbones"
            else "shared_siamese_efficientnet_b0_difference_head32"
        ),
        "model_variant": model_variant,
        "target": target,
        "model_variant": model_variant,
        "target_scaler": scaler,
        "selected_stage": selected_stage,
        "model_state_dict": selected_state,
        "pretrained_weight_path": str(weight_path),
        "seed": seed,
    }, run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "target": target,
        "device": str(device),
        "seed": seed,
        "parameters": total,
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "selected_stage": selected_stage,
        "scaler": scaler,
        "views": list(train_views),
        "frames_per_video": 20,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[job-complete] target={target} selected_stage={selected_stage} "
        f"test_MAE={metrics.loc[metrics.split.eq('test'), 'mae'].iloc[0]:.5g}",
        flush=True,
    )
    return metrics.to_dict("records")
