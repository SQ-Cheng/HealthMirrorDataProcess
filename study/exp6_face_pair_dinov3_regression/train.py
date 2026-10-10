"""Head-only weighted frame-pair regression with unchanged pair evaluation."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.exp2_face_dinov3_frozen.backbone import Head
from study.exp6_face_pair_lab_delta.train import _metrics, _weighted_loss, seed_everything
from . import config
from .data import loader
from .models import head_parameters


@torch.no_grad()
def evaluate(model, dataset, batches, device, split, scaler):
    model.eval()
    scores, rows, truths = [], [], []
    loss_sum = weight_sum = 0.
    for features, labels, indices, weights in batches:
        features, labels, weights = (value.to(device, non_blocking=True) for value in (features, labels, weights))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(features).squeeze(1)
        prediction = prediction.float()
        loss_sum += float(_weighted_loss(prediction, labels, weights).cpu()) * float(weights.sum().cpu())
        weight_sum += float(weights.sum().cpu())
        scores.append(prediction.cpu().numpy())
        truths.append(labels.cpu().numpy())
        rows.append(dataset.frame_pair_rows[indices.numpy()])
    frame = pd.DataFrame({"pair_row": np.concatenate(rows), "prediction": np.concatenate(scores),
                          "label": np.concatenate(truths)})
    aggregate = frame.groupby("pair_row", sort=True).agg(
        y_pred_scaled=("prediction", "mean"), frame_count=("prediction", "size"),
        y_true_scaled=("label", "first"), frame_prediction_std=("prediction", "std"))
    if len(aggregate) != len(dataset.records) or not aggregate.frame_count.eq(20).all():
        raise RuntimeError("Every pair must retain all twenty original frame pairs")
    prediction = dataset.records.iloc[aggregate.index].reset_index(drop=True).copy()
    np.testing.assert_array_equal(aggregate.y_true_scaled.to_numpy(np.float32), prediction.scaled_delta.to_numpy(np.float32))
    np.testing.assert_array_equal(prediction.raw_delta, prediction.second_value - prediction.first_value)
    for column in aggregate:
        prediction[column] = aggregate[column].to_numpy()
    prediction["y_true"] = prediction.raw_delta.to_numpy(np.float64)
    prediction["y_pred"] = prediction.y_pred_scaled * scaler["iqr"] + scaler["median"]
    prediction["split"] = split
    metrics = {**_metrics(prediction.y_true, prediction.y_pred), "loss": loss_sum / weight_sum}
    return metrics, prediction


def train_task(target, hidden, records, scaler, index, device, run_dir, provenance, epochs=None, cache_dir=config.CACHE,
               model_factory=None, loader_factory=None, expected_parameters=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED)
    torch.set_num_threads(1)
    groups = {split: records.loc[records.split.eq(split)].reset_index(drop=True) for split in ("train", "val", "test")}
    make_loader = loader_factory or loader
    augmented, training = make_loader(index, groups["train"], True, cache_dir)
    datasets, batches = {}, {}
    for split in groups:
        datasets[split], batches[split] = make_loader(index, groups[split], False, cache_dir)
    model = (model_factory or Head)(hidden).to(device)
    expected_parameters = head_parameters(hidden) if expected_parameters is None else expected_parameters
    if sum(p.numel() for p in model.parameters()) != expected_parameters:
        raise RuntimeError("Wrong head parameter count")
    epochs = config.MAX_EPOCHS if epochs is None else epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.MIN_LEARNING_RATE)
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    history, best_mae, patience, best = [], np.inf, 0, None
    print(f"[job-start] target={target} head={hidden} device={device} parameters={expected_parameters} "
          f"pairs={len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])} "
          "batch=12pairs*20framepairs*1view weighting=inverse_patient_pair_count", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.perf_counter()
        loss_sum = weight_sum = inputs = steps = 0
        for features, labels, _, weights in training:
            features, labels, weights = (value.to(device, non_blocking=True) for value in (features, labels, weights))
            optimizer.zero_grad(set_to_none=True)
            denominator = weights.sum()
            batch_loss = 0.
            for start in range(0, len(labels), config.MICROBATCH_FRAME_PAIRS):
                stop = start + config.MICROBATCH_FRAME_PAIRS
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    predicted = model(features[start:stop]).squeeze(1)
                    loss = _weighted_loss(predicted, labels[start:stop], weights[start:stop])
                    loss = loss * weights[start:stop].sum() / denominator
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite weighted frame-pair loss")
                amp.scale(loss).backward()
                batch_loss += float(loss.detach().cpu())
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            amp.step(optimizer)
            amp.update()
            batch_weight = float(denominator.cpu())
            loss_sum += batch_loss * batch_weight
            weight_sum += batch_weight
            inputs += len(labels)
            steps += 1
        if inputs != len(groups["train"]) * 100:
            raise RuntimeError("Dropped or duplicated frame-pair/view inputs")
        elapsed = time.perf_counter() - started
        train, _ = evaluate(model, datasets["train"], batches["train"], device, "train", scaler)
        val, _ = evaluate(model, datasets["val"], batches["val"], device, "val", scaler)
        history.append({"target": target, "head_hidden": hidden, "stage": "head", "stage_epoch": epoch,
                        "global_epoch": epoch, "train_optimization_loss": loss_sum / weight_sum,
                        **{f"train_{key}": value for key, value in train.items()},
                        **{f"val_{key}": value for key, value in val.items()},
                        "learning_rate": optimizer.param_groups[0]["lr"], "train_model_inputs": inputs,
                        "optimizer_steps": steps, "train_seconds": elapsed, "train_inputs_per_second": inputs / elapsed})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val["mae"] < best_mae - 1e-8:
            best_mae, patience = val["mae"], 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best, "target": target, "head_hidden": hidden,
                        "head_parameters": expected_parameters, "best_epoch": epoch, "seed": config.SEED,
                        "target_scaler": scaler, "backbone_frozen": True, "fusion": "late CLS minus early CLS",
                        "frames_per_video": 20, "loss_level": "frame_pair", "train_batch_policy": "distinct_lab_views",
                        "lab_pairs_per_batch": 12, "microbatch_frame_pairs": 120, **provenance}, run_dir / "model.pt")
        else:
            patience += 1
        print(f"[epoch] target={target} head={hidden} {epoch:03d}/{epochs} "
              f"train_loss={train['loss']:.4f} val_loss={val['loss']:.4f} train_MAE={train['mae']:.5g} "
              f"val_MAE={val['mae']:.5g} val_r={val['pearson_r']:.4f} val_R2={val['r2']:.4f} "
              f"val_dir_bACC={val['direction_balanced_accuracy']:.4f} lr={history[-1]['learning_rate']:.2e} "
              f"throughput={inputs / elapsed:.1f}/s patience={patience}/{config.PATIENCE}", flush=True)
        scheduler.step()
        if patience >= config.PATIENCE:
            break
    if best is None:
        raise RuntimeError("No finite validation-selected checkpoint")
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True)["model_state_dict"], strict=True)
    metrics, predictions = [], []
    for split in groups:
        result, predicted = evaluate(model, datasets[split], batches[split], device, split, scaler)
        metrics.append({"target": target, "head_hidden": hidden, "split": split, **result})
        predictions.append(predicted)
    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run_dir / "pair_predictions.csv", index=False)
    print(f"[job-complete] target={target} head={hidden} test_MAE={metrics[-1]['mae']:.5g}", flush=True)
