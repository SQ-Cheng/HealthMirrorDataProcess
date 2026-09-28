"""Two-stage R3D-18 training with video-level validation and test metrics."""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .clips import OneClipPerVideoSampler, VideoClipDataset
from .config import (
    EVAL_BATCH_SIZE, EVAL_WORKERS, FINETUNE_EPOCHS, FINETUNE_LR,
    FINETUNE_MIN_LR, FINETUNE_PATIENCE, GRAD_CLIP_NORM,
    HEAD_EPOCHS, HEAD_LR, HEAD_MIN_LR, HEAD_PATIENCE,
    KINETICS_MEAN, KINETICS_STD, PREFETCH_FACTOR,
    SMOOTH_L1_BETA, TRAIN_BATCH_SIZE, TRAIN_WORKERS, WEIGHT_DECAY,
)
from .models import build_model, freeze_encoder, unfreeze_all


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _loader(index, records, train):
    dataset = VideoClipDataset(index, records, train=train)
    workers = TRAIN_WORKERS if train else EVAL_WORKERS
    return dataset, DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE if train else EVAL_BATCH_SIZE,
        sampler=OneClipPerVideoSampler(dataset) if train else None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
    )


def _normalize(clips, device):
    clips = clips.to(device, non_blocking=True).float().div_(255.0)
    mean = clips.new_tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
    std = clips.new_tensor(KINETICS_STD).view(1, 3, 1, 1, 1)
    return (clips - mean) / std


def _metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "n": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan,
        "pearson_r": (
            float(np.corrcoef(truth, prediction)[0, 1])
            if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0
            else np.nan
        ),
        "spearman_r": float(pd.Series(truth).rank().corr(pd.Series(prediction).rank())),
    }


@torch.no_grad()
def _evaluate(model, dataset, loader, device, scaler, split):
    model.eval()
    predictions, video_rows = [], []
    for clips, _, rows in loader:
        inputs = _normalize(clips, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
        predictions.append(output.float().cpu().numpy())
        video_rows.append(rows.numpy())
    frame = pd.DataFrame({
        "video_row": np.concatenate(video_rows),
        "clip_pred_scaled": np.concatenate(predictions),
    }).groupby("video_row", as_index=False).agg(
        y_pred_scaled=("clip_pred_scaled", "mean"),
        clip_count=("clip_pred_scaled", "size"),
        clip_prediction_std=("clip_pred_scaled", "std"),
    )
    records = dataset.records.iloc[frame.video_row.to_numpy(int)].reset_index(drop=True)
    frame["y_true_scaled"] = records.robust_scaled_raw_value.to_numpy(float)
    frame["y_true"] = records.raw_value.to_numpy(float)
    frame["y_pred"] = frame.y_pred_scaled * scaler["iqr"] + scaler["median"]
    frame["hospital_id"] = records.hospital_id.to_numpy(str)
    frame["video_id"] = records.video_id.to_numpy(str)
    frame["source_sample_id"] = records.source_sample_id.to_numpy(str)
    frame["match_delta_h"] = records.match_delta_h.to_numpy(float)
    frame.insert(0, "split", split)
    result = _metrics(frame.y_true, frame.y_pred)
    result["loss"] = float(F.smooth_l1_loss(
        torch.from_numpy(frame.y_pred_scaled.to_numpy(np.float32)),
        torch.from_numpy(frame.y_true_scaled.to_numpy(np.float32)),
        beta=SMOOTH_L1_BETA,
    ))
    return result, frame.drop(columns="video_row")


def _train_epoch(model, loader, optimizer, amp_scaler, device, frozen):
    if frozen:
        model.eval()
        model.head.train()
    else:
        model.train()
    total_loss, video_count = 0.0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for clips, labels, _ in loader:
        inputs = _normalize(clips, device)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(inputs)
            loss = F.smooth_l1_loss(predictions, labels, beta=SMOOTH_L1_BETA)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        total_loss += float(loss.detach().cpu()) * len(labels)
        video_count += len(labels)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "train_loss": total_loss / max(video_count, 1),
        "train_videos": video_count,
        "train_seconds": seconds,
        "videos_per_second": video_count / max(seconds, 1e-8),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def _clone(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _stage(stage, model, datasets, loaders, target, device, run_dir, scaler,
           history, epochs, patience_limit, learning_rate, minimum_learning_rate):
    frozen = stage == "head"
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=minimum_learning_rate)
    amp_scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    best_state, best_mae, patience, offset = None, np.inf, 0, len(history)
    print(f"[stage-start] target={target} stage={stage} lr={learning_rate:.1e} "
          f"lr_min={minimum_learning_rate:.1e} epochs={epochs} "
          f"patience={patience_limit}", flush=True)
    for epoch in range(1, epochs + 1):
        step = _train_epoch(model, loaders["train_sampled"], optimizer,
                            amp_scaler, device, frozen)
        validation, _ = _evaluate(
            model, datasets["val"], loaders["val"], device, scaler, "val"
        )
        row = {
            "target": target, "stage": stage, "stage_epoch": epoch,
            "global_epoch": offset + epoch, **step,
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if validation["mae"] < best_mae - 1e-8:
            best_state, best_mae, patience, marker = (
                _clone(model), validation["mae"], 0, "*"
            )
        else:
            patience += 1
            marker = ""
        print(f"[epoch] target={target} stage={stage} {epoch:03d}/{epochs} "
              f"train_loss={step['train_loss']:.5f} "
              f"val_MAE={validation['mae']:.5f} "
              f"val_r={validation['pearson_r']:.4f} "
              f"throughput={step['videos_per_second']:.1f}/s "
              f"memory={step['peak_gpu_memory_gb']:.2f}GiB "
              f"patience={patience}/{patience_limit}{marker}", flush=True)
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] target={target} stage={stage}", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint: {target}/{stage}")
    return best_state, best_mae


def train_task(target, records, scaler, index, device_id, run_dir, seed):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed(seed)
    torch.cuda.set_device(device_id)
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{device_id}")
    split_records = {
        split: records.loc[records.split.eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    datasets, loaders = {}, {}
    datasets["train_sampled"], loaders["train_sampled"] = _loader(
        index, split_records["train"], True
    )
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(index, split_records[split], False)
    model, weight_sha256 = build_model()
    model = model.to(device)
    freeze_encoder(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_head = sum(parameter.numel() for parameter in model.parameters()
                         if parameter.requires_grad)
    print(f"[job-start] target={target} device={device} "
          f"videos={len(split_records['train'])}/{len(split_records['val'])}/"
          f"{len(split_records['test'])} "
          f"patients={split_records['train'].hospital_id.nunique()}/"
          f"{split_records['val'].hospital_id.nunique()}/"
          f"{split_records['test'].hospital_id.nunique()} "
          f"clips={len(datasets['train'])}/{len(datasets['val'])}/"
          f"{len(datasets['test'])} parameters={parameter_count} "
          f"head_trainable={trainable_head}", flush=True)
    history = []
    head_state, head_mae = _stage(
        "head", model, datasets, loaders, target, device, run_dir, scaler,
        history, HEAD_EPOCHS, HEAD_PATIENCE, HEAD_LR, HEAD_MIN_LR,
    )
    model.load_state_dict(head_state)
    unfreeze_all(model)
    fine_state, fine_mae = _stage(
        "finetune", model, datasets, loaders, target, device, run_dir, scaler,
        history, FINETUNE_EPOCHS, FINETUNE_PATIENCE, FINETUNE_LR, FINETUNE_MIN_LR,
    )
    selected_stage, selected_state = (
        ("finetune", fine_state) if fine_mae <= head_mae else ("head", head_state)
    )
    model.load_state_dict(selected_state)
    metrics, predictions = [], []
    for split in ("train", "val", "test"):
        values, frame = _evaluate(
            model, datasets[split], loaders[split], device, scaler, split
        )
        metrics.append({"target": target, "split": split,
                        "selected_stage": selected_stage, **values})
        predictions.append(frame)
    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    torch.save({
        "schema_version": 1, "experiment": "exp3_video_lab_regression",
        "architecture": "kinetics_r3d18_head32", "target": target,
        "model_state_dict": selected_state, "selected_stage": selected_stage,
        "target_scaler": scaler, "seed": seed,
        "pretrained_weight_sha256": weight_sha256,
    }, run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "target": target, "seed": seed, "selected_stage": selected_stage,
        "architecture": "kinetics_r3d18_head32",
        "head_parameters": trainable_head, "total_parameters": parameter_count,
        "scaler": scaler, "clip_frames": 16,
        "train_policy": "one random indexed clip per video each epoch",
        "eval_policy": "mean predictions over all indexed clips per video",
        "head_lr": HEAD_LR, "head_epochs": HEAD_EPOCHS,
        "head_patience": HEAD_PATIENCE,
        "finetune_lr": FINETUNE_LR, "finetune_epochs": FINETUNE_EPOCHS,
        "finetune_patience": FINETUNE_PATIENCE,
        "weight_decay": WEIGHT_DECAY,
    }, indent=2), encoding="utf-8")
    test = next(item for item in metrics if item["split"] == "test")
    print(f"[job-complete] target={target} selected_stage={selected_stage} "
          f"test_MAE={test['mae']:.5f} test_r={test['pearson_r']:.4f}", flush=True)
    for dataset in datasets.values():
        dataset.close()
    return metrics
