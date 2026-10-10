"""Single-stage fitting with the same twenty-frame, per-view supervised losses."""

import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.common.video_loss import VideoBCELoss, VideoSmoothL1Loss
from study.exp2_binary_classification_common.engine import _metrics as classification_metrics
from study.exp2_face_pretrained_head32_regression.train import _prepare_images, _regression_metrics

from . import config
from .data import loader, fit_feature_scaler
from .models import build_model


def prepare_inputs(inputs, views, architecture, device):
    if architecture == "small_cnn":
        return _prepare_images(inputs, views, "bicubic", device)
    return inputs.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, dataset, batches, architecture, family, target, criterion, device, scaler, split):
    model.eval()
    scores, rows = [], []
    loss_sum = loss_count = 0
    for inputs, labels, indices, views in batches:
        inputs = prepare_inputs(inputs, views, architecture, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(inputs).squeeze(1)
            loss = criterion(output, labels.to(device))
        probability_or_scaled = output.float().sigmoid() if family == "classification" else output.float()
        scores.append(probability_or_scaled.cpu().numpy())
        rows.append(dataset.frame_video_rows[indices.numpy()])
        loss_sum += float(loss.sum().cpu())
        loss_count += loss.numel()
    frame = pd.DataFrame({"video_row": np.concatenate(rows), "prediction": np.concatenate(scores)})
    pooled = frame.groupby("video_row", sort=True).prediction.agg(["mean", "size"])
    if len(pooled) != len(dataset.video_records) or not pooled["size"].eq(20).all():
        raise RuntimeError("Evaluation did not retain exactly twenty frames per video")
    records = dataset.video_records.iloc[pooled.index].copy().reset_index(drop=True)
    predicted = pooled["mean"].to_numpy(float)
    if family == "classification":
        metrics = classification_metrics(records.binary_label, predicted)
        predictions = records[["hospital_id", "video_id", "binary_label"]].copy()
        predictions["y_true"] = records.binary_label.astype(int)
        predictions["y_probability"] = predicted
        predictions["y_pred"] = (predicted >= .5).astype(int)
        predictions["input_count"] = 20
    else:
        raw = scaler.inverse_transform(predicted)
        metrics = _regression_metrics(records.raw_value, raw, records.score_threshold,
                                      config.reference.config.SCORE_DEFINITIONS[target]["direction"])
        predictions = records[["hospital_id", "video_id", "raw_value", "score_threshold"]].copy()
        predictions["y_true"] = records.raw_value
        predictions["y_pred"] = raw
        predictions["y_pred_scaled"] = predicted
        predictions["frame_count"] = 20
    predictions.insert(0, "split", split)
    metrics["loss"] = loss_sum / loss_count
    return metrics, predictions


def train_task(job, index, device, *, experiment_config=None, model_factory=None,
               loader_factory=None, feature_scaler=None):
    cfg = experiment_config or config
    make_model = model_factory or build_model
    make_loader = loader_factory or loader
    scale_features = feature_scaler or fit_feature_scaler
    architecture, family, target = job["architecture"], job["family"], job["target"]
    output = getattr(cfg, "OUTPUT_DIR", cfg.HERE / "outputs")
    root = output / family / architecture
    run = root / "runs" / target
    run.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(output / f"source_records/{target}.csv",
                          dtype={"hospital_id": str, "video_id": str})
    seed = cfg.reference.job_seed(family, target)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    groups = {split: records.loc[records.split.eq(split)].reset_index(drop=True)
              for split in ("train", "val", "test")}
    datasets, batches = {}, {}
    datasets["augmented"], batches["augmented"] = make_loader(index, groups["train"], architecture, family, True)
    for split, group in groups.items():
        datasets[split], batches[split] = make_loader(index, group, architecture, family, False)
    model = make_model(architecture)
    scale_features(model, index, records, architecture)
    model = model.to(device)
    if architecture == "small_cnn":
        model = model.to(memory_format=torch.channels_last)
    scaler = cfg.reference.RobustTargetScaler(**job["scaler"])
    positives = int(groups["train"].binary_label.sum())
    negatives = len(groups["train"]) - positives
    pos_weight = negatives / positives
    loss_level = getattr(cfg, "LOSS_LEVEL", "video_view")
    if loss_level == "frame":
        criterion = (nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
                     if family == "classification" else nn.SmoothL1Loss(beta=.5, reduction="none"))
    elif loss_level == "video_view":
        criterion = VideoBCELoss(torch.tensor(pos_weight, device=device)) if family == "classification" else VideoSmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss level: {loss_level}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATES[architecture], weight_decay=cfg.WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS,
                                                         eta_min=cfg.MIN_LEARNING_RATES[architecture])
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    history, best, patience, best_score = [], None, 0, -np.inf
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[job-start] {architecture}/{family}/{target} device={device} parameters={parameter_count} "
          f"videos={len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])} "
          f"batch=12labs*20frames*1view loss_unit={loss_level} pos_weight={pos_weight:.6g}", flush=True)
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        started = time.perf_counter()
        loss_sum = units = inputs_count = 0
        for inputs, labels, _, views in batches["augmented"]:
            inputs = prepare_inputs(inputs, views, architecture, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                per_view = criterion(model(inputs).squeeze(1), labels.to(device))
                loss = per_view.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite view loss; no samples silently skipped")
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            amp.step(optimizer)
            amp.update()
            loss_sum += float(per_view.detach().sum().cpu())
            units += per_view.numel()
            inputs_count += len(labels)
        if inputs_count != len(groups["train"]) * 100:
            raise RuntimeError("Training dropped or duplicated a frame/view")
        duration = time.perf_counter() - started
        train, _ = evaluate(model, datasets["train"], batches["train"], architecture, family, target, criterion, device, scaler, "train")
        val, _ = evaluate(model, datasets["val"], batches["val"], architecture, family, target, criterion, device, scaler, "val")
        score = val["balanced_accuracy"] if family == "classification" else -val["mae"]
        row = {"architecture": architecture, "family": family, "target": target, "stage": "joint",
               "epoch": epoch, "global_epoch": epoch, "train_optimization_loss": loss_sum / units,
               "learning_rate": optimizer.param_groups[0]["lr"], "train_model_inputs": inputs_count,
               "train_seconds": duration, "train_inputs_per_second": inputs_count / duration,
               **{f"train_{key}": value for key, value in train.items()},
               **{f"val_{key}": value for key, value in val.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        if score > best_score + 1e-4:
            best_score, patience = score, 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best, "architecture": architecture, "family": family, "target": target,
                        "seed": seed, "best_epoch": epoch, "parameters": parameter_count,
               "loss_unit": loss_level, "frames_per_view": 20, "pos_weight": pos_weight,
                        "target_scaler": scaler.to_dict() if family == "regression" else None}, run / "model.pt")
        else:
            patience += 1
        print(f"[epoch] {architecture}/{family}/{target} {epoch:03d}/{cfg.MAX_EPOCHS} "
              f"train_loss={train['loss']:.4f} val_loss={val['loss']:.4f} "
              f"val_score={score:.4f} lr={row['learning_rate']:.2e} "
              f"throughput={inputs_count / duration:.1f}/s patience={patience}/{cfg.PATIENCE}", flush=True)
        schedule.step()
        if patience >= cfg.PATIENCE:
            break
    if best is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best)
    metrics, predictions = [], []
    for split in groups:
        metric, prediction = evaluate(model, datasets[split], batches[split], architecture, family, target, criterion, device, scaler, split)
        metrics.append({"architecture": architecture, "target": target, "split": split, **metric})
        predictions.append(prediction.assign(architecture=architecture, target=target))
    pd.DataFrame(metrics).to_csv(run / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run / "video_predictions.csv", index=False)
    for dataset in datasets.values():
        if hasattr(dataset, "close"):
            dataset.close()
    print(f"[job-complete] {architecture}/{family}/{target}", flush=True)
