"""Raw-unit validation, leakage-safe fitting and bounded online accumulation."""

import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from study.exp2_face_pretrained_head32_regression.train import _regression_metrics
from . import config
from .data import read_records, fit_transform, inverse, train_weights, loader, fit_feature_stats
from .models import Fusion


def weighted_objective(prediction, labels, video_rows, weights, settings):
    if len(labels) % 20:
        raise ValueError("Incomplete video/view group")
    group_rows = video_rows.reshape(-1, 20)
    if not group_rows.eq(group_rows[:, :1]).all() or not labels.reshape(-1, 20).eq(labels.reshape(-1, 20)[:, :1]).all():
        raise ValueError("Mixed labels inside a video/view")
    truth = labels.reshape(-1, 20)[:, 0]
    pooled = prediction.float().reshape(-1, 20).mean(1)
    video_weight = weights[group_rows[:, 0]].to(prediction.device)
    if settings["loss"] == "frame":
        losses = F.smooth_l1_loss(prediction.float(), labels, beta=.5, reduction="none").reshape(-1, 20).mean(1)
    else:
        losses = F.smooth_l1_loss(pooled, truth, beta=.5, reduction="none")
    loss = (losses*video_weight).sum()/video_weight.sum()
    if settings["rank"]:
        # Signed continuous-target ranking; boundary pairs carry no ranking loss.
        delta = truth[:, None]-truth[None, :]
        mask = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1) & delta.abs().ge(.25)
        if mask.any():
            difference = pooled[:, None]-pooled[None, :]
            loss = loss+settings["rank"]*F.softplus(-delta.sign()[mask]*difference[mask]/.2).mean()
    return loss


@torch.no_grad()
def evaluate(model, dataset, batches, scaler, target, device, split, online=False):
    model.eval()
    scores = np.zeros(len(dataset.video_records), dtype=np.float64)
    count = np.zeros(len(scores), dtype=np.int64)
    loss_sum = 0.
    for features, labels, video, views, pixels in batches:
        if online:
            outputs = [model.forward_video(features[start:start+20], pixels[start:start+20], views[start:start+20], device)
                       for start in range(0, len(labels), 20)]
            prediction = torch.cat(outputs).float().cpu().numpy()
        else:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(features.to(device)).float().cpu().numpy()
        if not np.isfinite(prediction).all():
            raise RuntimeError("Nonfinite validation/test output")
        np.add.at(scores, video.numpy(), prediction)
        np.add.at(count, video.numpy(), 1)
        loss_sum += float(F.smooth_l1_loss(torch.from_numpy(prediction), labels, beta=.5, reduction="sum"))
    if not np.all(count == 20):
        raise RuntimeError("Evaluation must retain all twenty original frames/video")
    records = dataset.video_records
    predicted = inverse(scores/count, scaler)
    metrics = _regression_metrics(records.raw_value, predicted, records.score_threshold,
                                   config.reference.paired.SCORE_DEFINITIONS[target]["direction"])
    metrics["loss"] = loss_sum/count.sum()
    output = records[["hospital_id", "video_id", "raw_value", "score_threshold"]].copy()
    output = output.assign(split=split, y_true=records.raw_value, y_pred=predicted, y_pred_scaled=scores/count, frame_count=count)
    return metrics, output


def schedule_lr(optimizer, epoch, max_epochs, warmup, initial, floors):
    for group, lr, floor in zip(optimizer.param_groups, initial, floors):
        if epoch <= warmup:
            group["lr"] = lr*epoch/warmup
        else:
            progress = (epoch-warmup-1)/max(1, max_epochs-warmup-1)
            group["lr"] = floor+(lr-floor)*.5*(1+math.cos(math.pi*progress))


def train_epoch(model, batches, optimizer, amp, weights, settings, device, online=False):
    model.train()
    count = steps = batches_count = 0
    loss_sum = 0.
    for features, labels, video, views, pixels in batches:
        optimizer.zero_grad(set_to_none=True)
        if online:
            # Backpropagate one complete video at a time; one update per logical batch.
            rows = video.reshape(-1, 20)
            denominator = weights[rows[:, 0]].sum().to(device)
            objective = 0.
            for start in range(0, len(labels), 20):
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    prediction = model.forward_video(features[start:start+20], pixels[start:start+20], views[start:start+20], device)
                    loss = F.smooth_l1_loss(prediction.float().mean(), labels[start].to(device), beta=.5)
                    loss = loss*weights[video[start]].to(device)/denominator
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite online loss")
                amp.scale(loss).backward()
                objective += float(loss.detach())
        else:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(features.to(device))
                loss = weighted_objective(prediction, labels.to(device), video, weights, settings)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite cached-feature loss")
            amp.scale(loss).backward()
            objective = float(loss.detach())
        amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        before = amp.get_scale()
        amp.step(optimizer)
        amp.update()
        steps += int(amp.get_scale() >= before)
        count += len(labels)
        loss_sum += objective
        batches_count += 1
    if not steps:
        raise RuntimeError("No optimizer updates in this epoch")
    if count != len(batches.dataset):
        raise RuntimeError("A training frame/view was dropped or duplicated")
    return loss_sum/max(batches_count, 1), count, steps


def train_task(name, target, index, ir_index, device, manifest, warm_checkpoint=None):
    settings = config.get_variant(name)
    root = config.OUTPUT / name / "runs" / target
    root.mkdir(parents=True, exist_ok=True)
    seed = manifest["job_seeds"][target]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    records = read_records(target)
    groups = {split: records.loc[records.split.eq(split)].reset_index(drop=True) for split in ("train", "val", "test")}
    scaler = fit_transform(records, target, settings)
    weights, weight_policy = train_weights(groups["train"], settings)
    weights = torch.from_numpy(weights)
    model = Fusion(settings)
    if settings["normalize"]:
        fit_feature_stats(model, index, groups["train"])
    if settings["warm_start"]:
        checkpoint = torch.load(warm_checkpoint, map_location="cpu", weights_only=True)
        # This RGB checkpoint is raw robust regression, not log regression.
        model.warm_start_rgb(checkpoint, {"unit": scaler["unit"], "median": scaler["median"], "iqr": scaler["iqr"]})
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    (root / "weight_policy.json").write_text(json.dumps(weight_policy, indent=2)+"\n")
    groups["train"].assign(training_weight=weights.numpy()).to_csv(root / "training_weights.csv", index=False)
    history, best, best_score, best_epoch, best_stage, best_tail = [], None, np.inf, 0, "frozen", None
    datasets = []
    phases = ["frozen"]+(["ir_tail"] if settings["online_epochs"] else [])
    for phase in phases:
        online = phase == "ir_tail"
        if online:
            from .online import OnlineIR
            model.load_state_dict(best, strict=True)
            model = OnlineIR(model, settings, device)
            encoder_params = [p for name, p in model.named_parameters() if name.startswith(("dino.", "farl.")) and p.requires_grad]
            optimizer = torch.optim.AdamW([
                {"params": model.fusion.parameters(), "lr": settings["online_head_lr"], "weight_decay": settings["weight_decay"]},
                {"params": encoder_params, "lr": settings["online_encoder_lr"], "weight_decay": settings["online_encoder_weight_decay"]}])
            epochs, patience = settings["online_epochs"], settings["online_patience"]
            initial, floors = [settings["online_head_lr"], settings["online_encoder_lr"]], [settings["online_head_min_lr"], settings["online_encoder_min_lr"]]
            warmup = settings["online_warmup"]
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
            epochs, patience = settings["epochs"], settings["patience"]
            initial, floors, warmup = [settings["lr"]], [settings["min_lr"]], settings["warmup"]
        local = {}
        train_data, train_batches = loader(index, groups["train"], settings, scaler, True, online, ir_index)
        datasets.append(train_data)
        for split in groups:
            local[split] = loader(index, groups[split], settings, scaler, False, online, ir_index)
            datasets.append(local[split][0])
        amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.)
        if not online and settings["warm_start"]:
            # Preserve a useful supervised RGB initializer if every update worsens it.
            initial_train, _ = evaluate(model, *local["train"], scaler, target, device, "train")
            initial_val, _ = evaluate(model, *local["val"], scaler, target, device, "val")
            best_score = initial_val["mae"]
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(dict(target=target, architecture=name, stage="frozen", epoch=0, global_epoch=0,
                stage_epoch=0, train_optimization_loss=np.nan, train_model_inputs=0, optimizer_steps=0,
                train_seconds=0., learning_rate=0., **{f"train_{key}": value for key, value in initial_train.items()},
                **{f"val_{key}": value for key, value in initial_val.items()}))
            initial_checkpoint = {"state_dict": best, "ir_tail_state": None, "variant": name, "target": target,
                "seed": seed, "best_epoch": 0, "selected_stage": "frozen", "target_transform": scaler,
                "settings": settings, "fusion_parameters": parameter_count, "source_contract": manifest["contract"]}
            torch.save(initial_checkpoint, root / "model.pt")
            torch.save(initial_checkpoint, root / "stage_frozen_best.pt")
        phase_best, stale = best_score if settings["warm_start"] and not online else np.inf, 0
        print(f"[stage-start] {name}/{target} {phase} device={device} parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
              f"train/val/test={len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])} loss={settings['loss']}", flush=True)
        for epoch in range(1, epochs+1):
            schedule_lr(optimizer, epoch, epochs, warmup, initial, floors)
            started = time.perf_counter()
            optimization_loss, inputs, steps = train_epoch(model, train_batches, optimizer, amp, weights, settings, device, online)
            train, _ = evaluate(model, *local["train"], scaler, target, device, "train", online)
            val, _ = evaluate(model, *local["val"], scaler, target, device, "val", online)
            global_epoch = int(history[-1]["epoch"])+1 if history else 1
            row = dict(target=target, architecture=name, stage=phase, epoch=global_epoch, global_epoch=global_epoch,
                stage_epoch=epoch, train_optimization_loss=optimization_loss, train_model_inputs=inputs, optimizer_steps=steps,
                train_seconds=time.perf_counter()-started, learning_rate=optimizer.param_groups[0]["lr"],
                **{f"train_{key}": value for key, value in train.items()}, **{f"val_{key}": value for key, value in val.items()})
            history.append(row)
            pd.DataFrame(history).to_csv(root / "history.csv", index=False)
            if not np.isfinite(val["mae"]):
                raise RuntimeError("Nonfinite validation MAE")
            phase_improved = val["mae"] < phase_best-1e-4
            if phase_improved:
                phase_best, stale = val["mae"], 0
            else:
                stale += 1
            global_improved = val["mae"] < best_score-1e-4
            if global_improved or phase_improved:
                fusion = model.fusion if online else model
                candidate_state = {key: value.detach().cpu().clone() for key, value in fusion.state_dict().items()}
                candidate_tail = model.tail_state() if online else None
                checkpoint = {"state_dict": candidate_state, "ir_tail_state": candidate_tail, "variant": name, "target": target,
                    "seed": seed, "best_epoch": global_epoch, "selected_stage": phase, "target_transform": scaler,
                    "settings": settings, "fusion_parameters": parameter_count, "source_contract": manifest["contract"]}
                if phase_improved:
                    torch.save(checkpoint, root / f"stage_{phase}_best.pt")
            if global_improved:
                best_score, best_epoch, best_stage = val["mae"], global_epoch, phase
                best, best_tail = candidate_state, candidate_tail
                torch.save(checkpoint, root / "model.pt")
            print(f"[epoch] {name}/{target} {phase} {epoch}/{epochs} train_mae={train['mae']:.5g} val_mae={val['mae']:.5g} "
                  f"train_r={train['pearson_r']:.3f} val_r={val['pearson_r']:.3f} lr={row['learning_rate']:.2g} stale={stale}/{patience}", flush=True)
            if stale >= patience:
                break
        del train_batches, local
        for dataset in datasets:
            dataset.close()
        datasets.clear()
    if best is None:
        raise RuntimeError("No valid trained checkpoint")
    use_online = best_stage == "ir_tail"
    if settings["online_epochs"]:
        model.fusion.load_state_dict(best)
        if use_online:
            model.load_tail(best_tail)
        else:
            model = model.fusion
    else:
        model.load_state_dict(best)
    metrics, predictions = [], []
    for split, group in groups.items():
        dataset, batches = loader(index, group, settings, scaler, False, use_online, ir_index)
        result, prediction = evaluate(model, dataset, batches, scaler, target, device, split, use_online)
        metrics.append(dict(target=target, architecture=name, split=split, selected_stage=best_stage, **result))
        predictions.append(prediction.assign(target=target, architecture=name))
        dataset.close()
        del batches
    pd.DataFrame(metrics).to_csv(root / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(root / "video_predictions.csv", index=False)
    (root / "job_complete.json").write_text(json.dumps({"contract": manifest["contract"]})+"\n")
    print(f"[job-complete] {name}/{target} selected={best_stage} best_epoch={best_epoch}", flush=True)
