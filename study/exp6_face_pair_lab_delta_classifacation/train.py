"""Two-stage classification with the original Exp6 architecture and data loader."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from study.exp6_face_pair_lab_delta.config import (
    FINETUNE_LEARNING_RATE, FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE,
    GRAD_CLIP_NORM, HEAD_LEARNING_RATE, HEAD_MAX_EPOCHS, HEAD_PATIENCE,
    MIN_LEARNING_RATE, TRAIN_SOURCE_BATCH_SIZE, VIEWS, WEIGHT_DECAY,
)
from study.exp6_face_pair_lab_delta.models import (
    build_model, freeze_backbone, parameter_counts, unfreeze_all,
)
from study.exp6_face_pair_lab_delta.train import (
    _clone, _compile, _frame_predictions, _loader, _prepare, seed_everything,
)


def _metrics(labels, logits, pos_weight):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    predicted = probabilities >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "loss": float(F.binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(labels).float(),
            pos_weight=torch.tensor(pos_weight),
        )),
        "accuracy": float((labels == predicted).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _evaluate(model, dataset, loader, device, split, pos_weight, max_batches=None):
    labels, logits, rows = _frame_predictions(model, loader, device, max_batches)
    aggregate = pd.DataFrame({
        "pair_row": dataset.frame_pair_rows[rows],
        "label_up": labels.astype(np.int64),
        "frame_logit": logits,
    }).groupby("pair_row", as_index=False).agg(
        label_up=("label_up", "first"),
        mean_logit=("frame_logit", "mean"),
        frame_count=("frame_logit", "size"),
        frame_logit_std=("frame_logit", "std"),
    )
    info = dataset.records.iloc[aggregate.pair_row.to_numpy(int)][[
        "pair_id", "target", "hospital_id", "first_video_id", "second_video_id",
        "first_value", "second_value", "raw_delta", "lab_interval_h",
        "first_match_delta_h", "second_match_delta_h",
    ]].reset_index(drop=True)
    result = pd.concat([info, aggregate.drop(columns="pair_row")], axis=1)
    result.insert(0, "split", split)
    result["probability_up"] = torch.sigmoid(
        torch.from_numpy(result.mean_logit.to_numpy(np.float32))
    ).numpy()
    result["predicted_up"] = (result.probability_up >= 0.5).astype(int)
    return _metrics(result.label_up, result.mean_logit, pos_weight), result


def _train_epoch(model, raw_model, loader, optimizer, scaler, device,
                 frozen, pos_weight):
    if frozen:
        model.eval()
        raw_model.head.train()
    else:
        model.train()
    weighted_sum = weight_total = input_count = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for first, second, target, _, codes, weights in loader:
        first = _prepare(first, codes, device)
        second = _prepare(second, codes, device)
        repeat = codes.shape[1]
        target = target.repeat_interleave(repeat).to(device, non_blocking=True)
        weights = weights.repeat_interleave(repeat).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(first, second).squeeze(1)
            values = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight, reduction="none"
            )
            loss = (values * weights).sum() / weights.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite classification loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        batch_weight = float(weights.sum().detach().cpu())
        weighted_sum += float(loss.detach().cpu()) * batch_weight
        weight_total += batch_weight
        input_count += len(target)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "train_optimization_loss": weighted_sum / max(weight_total, 1e-8),
        "train_model_inputs": int(input_count),
        "train_seconds": seconds,
        "train_inputs_per_second": input_count / max(seconds, 1e-8),
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def _stage(stage, model, datasets, loaders, target, device, run_dir,
           pos_weight, history, epochs, patience_limit, optimizer):
    execution, backend = _compile(model, target, stage)
    amp_scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE)
    best_state, best_loss, patience, offset = None, np.inf, 0, len(history)
    for epoch in range(1, epochs + 1):
        step = _train_epoch(
            execution, model, loaders["train_augmented"], optimizer,
            amp_scaler, device, stage == "head", pos_weight,
        )
        train_metrics, _ = _evaluate(
            execution, datasets["train"], loaders["train"], device,
            "train", float(pos_weight),
        )
        val_metrics, _ = _evaluate(
            execution, datasets["val"], loaders["val"], device,
            "val", float(pos_weight),
        )
        row = {
            "target": target, "stage": stage, "stage_epoch": epoch,
            "global_epoch": offset + epoch, **step,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "execution_backend": backend,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_metrics["loss"] < best_loss - 1e-8:
            best_loss, best_state, patience, marker = (
                val_metrics["loss"], _clone(model), 0, "*"
            )
        else:
            patience += 1
            marker = ""
        print(
            f"[epoch] target={target} stage={stage} {epoch:03d}/{epochs} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_bACC={val_metrics['balanced_accuracy']:.4f} "
            f"val_AUC={val_metrics['roc_auc']:.4f} "
            f"throughput={step['train_inputs_per_second']:.1f}/s "
            f"mem={step['peak_gpu_memory_gb']:.2f}GiB "
            f"patience={patience}/{patience_limit}{marker}", flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] target={target} stage={stage}", flush=True)
            break
    del execution
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {target}/{stage}")
    return best_state, best_loss


def train_task(target, records, frame_index, device_id, run_dir, seed):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.cuda.set_device(device_id)
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{device_id}")
    records = records.assign(scaled_delta=records.label_up.astype(np.float32))
    split_records = {
        name: records.loc[records.split.eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    for split, frame in split_records.items():
        if set(frame.label_up.unique()) != {0, 1}:
            raise ValueError(f"Both direction classes are required in {target}/{split}")
    train_neg = int(split_records["train"].label_up.eq(0).sum())
    train_pos = int(split_records["train"].label_up.eq(1).sum())
    pos_weight_value = train_neg / train_pos
    pos_weight = torch.tensor(pos_weight_value, device=device, dtype=torch.float32)
    train_augmented, train_augmented_loader = _loader(
        frame_index, split_records["train"], True, VIEWS, train_batch_policy="chunk"
    )
    datasets, loaders = {}, {"train_augmented": train_augmented_loader}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(
            frame_index, split_records[split], False
        )
    model, weight_path = build_model("shared")
    model = model.to(device, memory_format=torch.channels_last)
    freeze_backbone(model)
    total, head_trainable = parameter_counts(model)
    print(
        f"[job-start] target={target} device={device} "
        f"pairs={len(split_records['train'])}/{len(split_records['val'])}/"
        f"{len(split_records['test'])} "
        f"patients={split_records['train'].hospital_id.nunique()}/"
        f"{split_records['val'].hospital_id.nunique()}/"
        f"{split_records['test'].hospital_id.nunique()} "
        f"train_neg/pos={train_neg}/{train_pos} pos_weight={pos_weight_value:.5f} "
        f"train_inputs={train_augmented.model_input_count} "
        f"source_batch={TRAIN_SOURCE_BATCH_SIZE} "
        f"effective_pair_batch={TRAIN_SOURCE_BATCH_SIZE * len(VIEWS)} "
        f"batch_policy=chunk parameters={total} head_trainable={head_trainable}",
        flush=True,
    )
    history = []
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    print(
        f"[stage-start] target={target} stage=head lr={HEAD_LEARNING_RATE:.1e} "
        f"epochs={HEAD_MAX_EPOCHS} patience={HEAD_PATIENCE}", flush=True,
    )
    head_state, head_loss = _stage(
        "head", model, datasets, loaders, target, device, run_dir,
        pos_weight, history, HEAD_MAX_EPOCHS, HEAD_PATIENCE, optimizer,
    )
    model.load_state_dict(head_state)
    unfreeze_all(model)
    optimizer = AdamW(
        model.parameters(), lr=FINETUNE_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    print(
        f"[stage-start] target={target} stage=finetune lr={FINETUNE_LEARNING_RATE:.1e} "
        f"epochs={FINETUNE_MAX_EPOCHS} patience={FINETUNE_PATIENCE}", flush=True,
    )
    fine_state, fine_loss = _stage(
        "finetune", model, datasets, loaders, target, device, run_dir,
        pos_weight, history, FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, optimizer,
    )
    selected_stage, selected_state = (
        ("finetune", fine_state) if fine_loss <= head_loss else ("head", head_state)
    )
    model.load_state_dict(selected_state)
    metrics, predictions = [], []
    for split in ("train", "val", "test"):
        values, frame = _evaluate(
            model, datasets[split], loaders[split], device, split,
            pos_weight_value,
        )
        metrics.append({"target": target, "split": split,
                        "selected_stage": selected_stage,
                        "pos_weight": pos_weight_value, **values})
        predictions.append(frame)
    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        run_dir / "pair_predictions.csv", index=False
    )
    torch.save({
        "schema_version": 1,
        "experiment": "exp6_face_pair_lab_delta_classifacation",
        "architecture": "shared_siamese_efficientnet_b0_difference_head32",
        "target": target, "task_type": "binary_direction_classification",
        "model_state_dict": selected_state,
        "selected_stage": selected_stage,
        "pretrained_weight_path": str(weight_path),
        "seed": seed, "pos_weight": pos_weight_value,
        "decision_threshold": 0.5,
    }, run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "target": target, "seed": seed, "selected_stage": selected_stage,
        "train_neg": train_neg, "train_pos": train_pos,
        "pos_weight": pos_weight_value,
        "tie_policy": "exclude delta values numerically equal to zero (atol=1e-12)",
        "frame_count_per_pair": 20, "train_views": list(VIEWS),
        "batch_policy": "chunk", "source_batch_size": TRAIN_SOURCE_BATCH_SIZE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "finetune_learning_rate": FINETUNE_LEARNING_RATE,
        "head_max_epochs": HEAD_MAX_EPOCHS,
        "finetune_max_epochs": FINETUNE_MAX_EPOCHS,
        "head_patience": HEAD_PATIENCE,
        "finetune_patience": FINETUNE_PATIENCE,
        "min_learning_rate": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }, indent=2), encoding="utf-8")
    score = next(row["roc_auc"] for row in metrics if row["split"] == "test")
    print(f"[job-complete] target={target} selected_stage={selected_stage} "
          f"test_AUC={score:.4f}", flush=True)
    return metrics
