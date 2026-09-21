"""Four-GPU two-stage training and video-level partial-CASUS evaluation."""

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
    CASUS_MAX_SCORE, CROP_SCALE, EVAL_BATCH_SIZE, EVAL_NUM_WORKERS,
    FINETUNE_BACKBONE_LEARNING_RATE, FINETUNE_HEAD_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, GRAD_CLIP_NORM,
    HEAD_LEARNING_RATE, HEAD_MAX_EPOCHS, HEAD_PATIENCE, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, MIN_LEARNING_RATE, PREFETCH_FACTOR,
    SMOOTH_L1_BETA, TRAIN_NUM_WORKERS, TRAIN_SOURCE_BATCH_SIZE,
    TRAIN_VIEWS, WEIGHT_DECAY,
)
from .data import PairedCasusDataset, PostOnlyCasusDataset
from .models import (
    build_model, build_post_only_model, freeze_backbone, head_parameters,
    last_stage_parameters, parameter_counts, train_head_modules,
    unfreeze_last_stage,
)


PROTOCOLS = ("paired", "post_face_only")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(index, records, train, protocol="paired"):
    dataset_class = (
        PairedCasusDataset if protocol == "paired" else PostOnlyCasusDataset
    )
    dataset = dataset_class(
        index, records, TRAIN_VIEWS if train else ("original",), train,
    )
    workers = TRAIN_NUM_WORKERS if train else EVAL_NUM_WORKERS
    return dataset, DataLoader(
        dataset,
        batch_size=TRAIN_SOURCE_BATCH_SIZE if train else EVAL_BATCH_SIZE,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
    )


def _prepare(images, codes, device):
    images = images.to(device, non_blocking=True).float().div_(255)
    codes = codes.to(device, non_blocking=True)
    if codes.ndim == 2:
        images = images.repeat_interleave(codes.shape[1], dim=0)
        codes = codes.flatten()
    flip = codes.eq(1)
    if flip.any():
        images[flip] = torch.flip(images[flip], dims=(-1,))
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
        h, w = round(height * CROP_SCALE), round(width * CROP_SCALE)
        top, left = (height - h) // 2, (width - w) // 2
        output[crop] = F.interpolate(
            images[crop, :, top:top + h, left:left + w],
            size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
    mean = output.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = output.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((output - mean) / std).contiguous()


def _loss(prediction, target, weights):
    values = F.smooth_l1_loss(
        prediction, target, beta=SMOOTH_L1_BETA, reduction="none"
    )
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _train_epoch(model, core, loader, optimizer, scaler, device, gpu_ids, frozen, protocol):
    model.eval() if frozen else model.train()
    if frozen:
        train_head_modules(core)
    for gpu_id in gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu_id)
    total = weight_total = inputs = 0.0
    started = time.perf_counter()
    for batch in loader:
        if protocol == "paired":
            pre, post, target, _, codes, weights = batch
            model_inputs = (_prepare(pre, codes, device), _prepare(post, codes, device))
        else:
            post, target, _, codes, weights = batch
            model_inputs = (_prepare(post, codes, device),)
        repeat = codes.shape[1]
        target = target.repeat_interleave(repeat).to(device, non_blocking=True)
        weights = weights.repeat_interleave(repeat).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(*model_inputs).squeeze(1)
            loss = _loss(prediction, target, weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(core.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        batch_weight = float(weights.sum().detach().cpu())
        total += float(loss.detach().cpu()) * batch_weight
        weight_total += batch_weight
        inputs += len(target)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    memory = [torch.cuda.max_memory_allocated(gpu_id) / 1024**3 for gpu_id in gpu_ids]
    return {
        "loss": total / max(weight_total, 1e-8),
        "inputs": int(inputs),
        "seconds": elapsed,
        "throughput": inputs / max(elapsed, 1e-8),
        "memory_gb": memory,
    }


@torch.no_grad()
def _frame_predictions(model, loader, device, protocol):
    model.eval()
    truth, prediction, rows = [], [], []
    for batch in loader:
        if protocol == "paired":
            pre, post, target, frame_rows, codes, _ = batch
            model_inputs = (_prepare(pre, codes, device), _prepare(post, codes, device))
        else:
            post, target, frame_rows, codes, _ = batch
            model_inputs = (_prepare(post, codes, device),)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(*model_inputs).squeeze(1)
        truth.append(target.numpy() * CASUS_MAX_SCORE)
        prediction.append(output.float().cpu().numpy() * CASUS_MAX_SCORE)
        rows.append(frame_rows.numpy())
    return np.concatenate(truth), np.concatenate(prediction), np.concatenate(rows)


def _metrics(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    normalized_error = np.abs(prediction - truth) / CASUS_MAX_SCORE
    loss = np.where(
        normalized_error < SMOOTH_L1_BETA,
        0.5 * normalized_error**2 / SMOOTH_L1_BETA,
        normalized_error - 0.5 * SMOOTH_L1_BETA,
    ).mean()
    return {
        "n": len(truth),
        "loss": float(loss),
        "mae_points": float(mean_absolute_error(truth, prediction)),
        "rmse_points": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "explained_variance": float(explained_variance_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "pearson_r": float(np.corrcoef(truth, prediction)[0, 1]) if len(truth) > 1 and np.std(prediction) > 0 else np.nan,
        "spearman_r": float(pd.Series(truth).rank().corr(pd.Series(prediction).rank())) if len(truth) > 1 else np.nan,
    }


def _evaluate(model, dataset, loader, device, split, protocol):
    truth, prediction, rows = _frame_predictions(model, loader, device, protocol)
    video_rows = dataset.frame_video_rows[rows]
    aggregate = pd.DataFrame({
        "video_row": video_rows, "y_true": truth, "y_pred": prediction,
    }).groupby("video_row", as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        frame_count=("y_pred", "size"), frame_prediction_std=("y_pred", "std"),
    )
    info_columns = [
        "hospital_id", "video_id", "casus_score", "postoperative_progress",
    ]
    if protocol == "paired":
        info_columns.insert(1, "pre_video_id")
    info = dataset.records.iloc[
        aggregate.video_row.to_numpy(int)
    ][info_columns].reset_index(drop=True)
    result = pd.concat([info, aggregate.drop(columns="video_row")], axis=1)
    result.insert(0, "split", split)
    return _metrics(result.y_true, result.y_pred), result


def _clone(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _plot_history(history, path, protocol="paired"):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (
        ("loss", "Normalized SmoothL1 loss"),
        ("mae_points", "Video MAE (CASUS points)"),
        ("r2", "Video R2"),
        ("pearson_r", "Video Pearson r"),
    )
    for axis, (metric, title) in zip(axes.flat, panels):
        for stage, group in frame.groupby("stage", sort=False):
            axis.plot(group.global_epoch, group[f"train_{metric}"], label=f"{stage} train")
            axis.plot(group.global_epoch, group[f"val_{metric}"], "--", label=f"{stage} val")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    title = "Paired pre/post" if protocol == "paired" else "Post-face-only"
    figure.suptitle(f"Exp5 {title} partial-CASUS training")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _stage(stage, core, model, datasets, loaders, device, gpu_ids, run_dir, history, epochs, patience_limit, optimizer, protocol):
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE)
    best_state, best_mae, patience, offset = None, np.inf, 0, len(history)
    for epoch in range(1, epochs + 1):
        step = _train_epoch(
            model, core, loaders["train_augmented"], optimizer, scaler,
            device, gpu_ids, stage == "head", protocol,
        )
        train_metrics, _ = _evaluate(model, datasets["train"], loaders["train"], device, "train", protocol)
        val_metrics, _ = _evaluate(model, datasets["val"], loaders["val"], device, "val", protocol)
        row = {
            "stage": stage,
            "stage_epoch": epoch,
            "global_epoch": offset + epoch,
            "train_optimization_loss": step["loss"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items() if key not in {"n", "loss"}},
            **{f"val_{key}": value for key, value in val_metrics.items() if key not in {"n", "loss"}},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_model_inputs": step["inputs"],
            "train_seconds": step["seconds"],
            "train_inputs_per_second": step["throughput"],
            "peak_gpu_memory_all_gb": json.dumps([round(value, 3) for value in step["memory_gb"]]),
            "execution_backend": "torch.nn.DataParallel" if len(gpu_ids) > 1 else "eager",
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_metrics["mae_points"] < best_mae - 1e-4:
            best_mae, best_state, patience, marker = val_metrics["mae_points"], _clone(core), 0, "*"
        else:
            patience += 1
            marker = ""
        print(
            f"[epoch] stage={stage} {epoch:03d}/{epochs} train_loss={step['loss']:.5f} "
            f"train_MAE={train_metrics['mae_points']:.3f} train_R2={train_metrics['r2']:.4f} "
            f"val_MAE={val_metrics['mae_points']:.3f} val_RMSE={val_metrics['rmse_points']:.3f} "
            f"val_R2={val_metrics['r2']:.4f} val_r={val_metrics['pearson_r']:.4f} "
            f"throughput={step['throughput']:.1f}/s mem={','.join(f'{x:.2f}' for x in step['memory_gb'])}GiB "
            f"patience={patience}/{patience_limit}{marker}",
            flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] stage={stage}", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint in {stage}")
    return best_state, best_mae


def train(records, frame_index, seed, run_dir, protocol="paired"):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.set_num_threads(4)
    gpu_ids = list(range(torch.cuda.device_count()))
    if not gpu_ids:
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    split_records = {
        name: records[records.split.eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    augmented, augmented_loader = _loader(
        frame_index, split_records["train"], True, protocol,
    )
    datasets, loaders = {}, {"train_augmented": augmented_loader}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(
            frame_index, split_records[split], False, protocol,
        )

    core, weight_path = (
        build_model() if protocol == "paired" else build_post_only_model()
    )
    core = core.to(device)
    freeze_backbone(core)
    total, trainable = parameter_counts(core)
    model = nn.DataParallel(core, device_ids=gpu_ids, output_device=0) if len(gpu_ids) > 1 else core
    print(
        f"[job-start] protocol={protocol} gpus={gpu_ids} videos={len(split_records['train'])}/{len(split_records['val'])}/{len(split_records['test'])} "
        f"patients={split_records['train'].hospital_id.nunique()}/{split_records['val'].hospital_id.nunique()}/{split_records['test'].hospital_id.nunique()} "
        f"pairs={len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} "
        f"train_inputs={augmented.model_input_count} parameters={total} head_trainable={trainable}",
        flush=True,
    )
    history = []
    print(f"[stage-start] stage=head lr={HEAD_LEARNING_RATE:.1e} trainable={trainable}", flush=True)
    optimizer = AdamW(
        [parameter for parameter in core.parameters() if parameter.requires_grad],
        lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    head_state, head_mae = _stage(
        "head", core, model, datasets, loaders, device, gpu_ids, run_dir, history,
        HEAD_MAX_EPOCHS, HEAD_PATIENCE, optimizer, protocol,
    )
    core.load_state_dict(head_state)
    unfreeze_last_stage(core)
    _, trainable = parameter_counts(core)
    print(
        f"[stage-start] stage=last_stage backbone_lr={FINETUNE_BACKBONE_LEARNING_RATE:.1e} "
        f"head_lr={FINETUNE_HEAD_LEARNING_RATE:.1e} trainable={trainable}",
        flush=True,
    )
    optimizer = AdamW([
        {"params": last_stage_parameters(core), "lr": FINETUNE_BACKBONE_LEARNING_RATE},
        {"params": head_parameters(core), "lr": FINETUNE_HEAD_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    fine_state, fine_mae = _stage(
        "last_stage", core, model, datasets, loaders, device, gpu_ids, run_dir, history,
        FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, optimizer, protocol,
    )
    selected, state = ("last_stage", fine_state) if fine_mae <= head_mae else ("head", head_state)
    core.load_state_dict(state)
    metric_rows, predictions = [], []
    for split in ("train", "val", "test"):
        metrics, prediction = _evaluate(
            model, datasets[split], loaders[split], device, split, protocol,
        )
        metric_rows.append({"protocol": protocol, "seed": seed, "selected_stage": selected, "split": split, **metrics})
        predictions.append(prediction)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "video_predictions.csv", index=False)
    torch.save({
        "schema_version": 1,
        "seed": seed,
        "protocol": protocol,
        "architecture": (
            "independent_dual_efficientnet_b0_pair"
            if protocol == "paired"
            else "single_efficientnet_b0_post_face_capacity_controlled_fusion"
        ),
        "target": "four-laboratory partial-CASUS subtotal (0-16)",
        "selected_stage": selected,
        "state_dict": state,
        "pretrained_weight_path": str(weight_path),
    }, run_dir / "model.pt")
    _plot_history(history, run_dir / "training_history.png", protocol)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "seed": seed,
        "selected_stage": selected,
        "head_best_val_mae_points": head_mae,
        "last_stage_best_val_mae_points": fine_mae,
        "gpu_ids": gpu_ids,
        "parallel_backend": "torch.nn.DataParallel" if len(gpu_ids) > 1 else "single GPU",
        "train_views": list(TRAIN_VIEWS),
        "protocol": protocol,
        "input": (
            "one preoperative plus one postoperative RGB face"
            if protocol == "paired" else "one postoperative RGB face"
        ),
        "encoders": (
            "independent preoperative and postoperative EfficientNet-B0 plus projectors"
            if protocol == "paired" else "one postoperative EfficientNet-B0 plus projector"
        ),
        "missing_preoperative_branch": (
            None if protocol == "paired" else "fixed zero 64-dimensional embedding"
        ),
        "target_training_scale": "partial CASUS divided by 16",
        "evaluation": (
            "mean of 20 deterministic pre/post frame-pair predictions, reported in CASUS points"
            if protocol == "paired"
            else "mean of 20 deterministic postoperative-frame predictions, reported in CASUS points"
        ),
    }, indent=2), encoding="utf-8")
    test = metric_rows[-1]
    print(
        f"[job-complete] selected={selected} test_MAE={test['mae_points']:.3f}points "
        f"test_R2={test['r2']:.4f} test_r={test['pearson_r']:.4f}",
        flush=True,
    )
