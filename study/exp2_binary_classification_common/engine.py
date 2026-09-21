"""Train true binary classifiers while reusing the matched regression cohorts."""

from __future__ import annotations

import json
import math
import os
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from study.exp2_face_history_head32_regression import config as base_config
from study.exp2_face_history_head32_regression import models as face_history_models
from study.exp2_face_history_head32_regression import train as face_history_train
from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_history_head32_regression.history_data import HistoryFeatureStore
from study.exp2_face_pretrained_head32_regression import models as face_models
from study.exp2_face_pretrained_head32_regression import train as face_train
from study.exp2_history_only_head32_regression.data import HistoryOnlyDataset
from study.exp2_history_only_head32_regression.models import HistoryOnlyRegressor


MODALITIES = ("face_history", "face_only", "history_only")
TARGETS = (
    "oxyhemoglobin_fraction",
    "lactate_high",
    "urea_high",
    "total_bilirubin_high",
    "platelet_count_low",
    "hemoglobin_low",
    "aa_po2_ratio_low",
    "creatinine_high",
)
ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "study/exp2_face_history_head32_regression/outputs/20frame"
PREPARED_DIR = ROOT / "study/exp2_binary_classification_common/prepared"
FRAME_INDEX_PATH = PREPARED_DIR / "20frame_index/frame_offsets.npz"
WEIGHTS_DIR = ROOT / "study/exp2_face_pretrained/pretrained_weights"
EXPERIMENT_DIRS = {
    "face_history": ROOT / "study/exp2_face_history_head32_classification",
    "face_only": ROOT / "study/exp2_face_pretrained_head32_classification",
    "history_only": ROOT / "study/exp2_history_only_head32_classification",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_task(target: str):
    task_dir = PREPARED_DIR if target == "total_bilirubin_high" else REFERENCE_DIR
    records = pd.read_csv(
        task_dir / "task_records" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    required = {"hospital_id", "video_id", "binary_label", "split"}
    missing = required - set(records)
    if missing:
        raise ValueError(f"Missing classification columns for {target}: {sorted(missing)}")
    labels = pd.to_numeric(records.binary_label, errors="raise")
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"{target} does not contain both binary classes")
    if records.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError(f"Patient leakage in reference split for {target}")
    if records.video_id.duplicated().any():
        raise AssertionError(f"Duplicate videos in reference split for {target}")
    records["binary_label"] = labels.astype(np.float32)
    history = HistoryFeatureStore.load(
        task_dir / "history_records" / f"{target}.npz"
    )
    if set(records.video_id.astype(str)) != set(history.video_ids.astype(str)):
        raise AssertionError(f"History/video mismatch for {target}")
    return records, history


def _set_binary_labels(dataset) -> None:
    labels = dataset.video_records.binary_label.to_numpy(np.float32)
    dataset.labels_by_video = labels
    if hasattr(dataset, "weights_by_video"):
        dataset.weights_by_video = np.ones(len(labels), dtype=np.float32)


def _image_loader(modality, frame_index, records, history, train):
    views = base_config.VIEW_NAMES if train else ("original",)
    if modality == "face_history":
        dataset, loader = face_history_train._loader(
            frame_index, records, history, views, "efficientnet_b0", train
        )
    else:
        dataset, loader = face_train._loader(
            frame_index, records, views, "efficientnet_b0", train
        )
    _set_binary_labels(dataset)
    return dataset, loader


def _history_loader(records, history, train, seed):
    dataset = HistoryOnlyDataset(records, history)
    dataset.labels = dataset.records.binary_label.to_numpy(np.float32)
    generator = torch.Generator().manual_seed(seed) if train else None
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=train,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def _build_model(modality):
    if modality == "face_history":
        model, head, weight_path = face_history_models.build_pretrained_model(
            "efficientnet_b0", str(WEIGHTS_DIR)
        )
        return model, head, weight_path
    if modality == "face_only":
        model, head, weight_path = face_models.build_pretrained_model(
            "efficientnet_b0", str(WEIGHTS_DIR)
        )
        return model, head, weight_path
    model = HistoryOnlyRegressor()
    return model, model.regressor, None


def _set_stage(model, head, modality, stage):
    if modality == "history_only":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    helpers = face_history_models if modality == "face_history" else face_models
    if stage == "head":
        helpers.freeze_encoder(model, head)
    else:
        helpers.unfreeze_all(model)


def _prepare_batch(batch, modality, device, training):
    if modality == "face_history":
        images, labels, rows, view_codes, history, mask, _ = batch
        images = face_history_train._prepare_images(images, view_codes, "bicubic", device)
        if training and view_codes.ndim == 2:
            repeat = view_codes.shape[1]
            labels = labels.repeat_interleave(repeat)
            history = history.repeat_interleave(repeat, dim=0)
            mask = mask.repeat_interleave(repeat, dim=0)
        inputs = (
            images,
            history.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
        )
    elif modality == "face_only":
        images, labels, rows, view_codes = batch
        images = face_train._prepare_images(images, view_codes, "bicubic", device)
        if training and view_codes.ndim == 2:
            labels = labels.repeat_interleave(view_codes.shape[1])
        inputs = (images,)
    else:
        history, mask, labels, rows = batch
        inputs = (
            history.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
        )
    return inputs, labels.to(device, non_blocking=True), rows


def _metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result = {
        "n": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    return result


@torch.no_grad()
def _evaluate(model, loader, dataset, modality, criterion, device, split):
    model.eval()
    labels_out, probabilities_out, rows_out = [], [], []
    loss_sum = count = 0
    for batch in loader:
        inputs, labels, rows = _prepare_batch(batch, modality, device, False)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(*inputs).squeeze(1)
            loss = criterion(logits, labels).mean()
        probabilities = torch.sigmoid(logits)
        labels_out.append(labels.cpu().numpy())
        probabilities_out.append(probabilities.float().cpu().numpy())
        rows_out.append(rows.numpy())
        loss_sum += float(loss.cpu()) * len(labels)
        count += len(labels)
    labels = np.concatenate(labels_out)
    probabilities = np.concatenate(probabilities_out)
    rows = np.concatenate(rows_out).astype(np.int64)
    if modality == "history_only":
        video_rows = rows
    else:
        video_rows = dataset.frame_video_rows[rows]
    aggregate = pd.DataFrame({
        "video_row": video_rows, "label": labels, "probability": probabilities,
    }).groupby("video_row", as_index=False).agg(
        label=("label", "first"), probability=("probability", "mean"),
        input_count=("probability", "size"),
    )
    records = (
        dataset.records if modality == "history_only" else dataset.video_records
    ).iloc[aggregate.video_row.to_numpy(np.int64)].reset_index(drop=True)
    predictions = records[["hospital_id", "video_id", "binary_label"]].copy()
    predictions.insert(0, "split", split)
    predictions["y_true"] = aggregate.label.astype(np.uint8)
    predictions["y_probability"] = aggregate.probability
    predictions["y_pred"] = (aggregate.probability >= 0.5).astype(np.uint8)
    predictions["input_count"] = aggregate.input_count.astype(np.int64)
    metrics = _metrics(predictions.y_true, predictions.y_probability)
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics, predictions


def _train_epoch(model, loader, modality, optimizer, scaler, criterion, device, frozen):
    model.eval() if frozen else model.train()
    if frozen:
        raw = getattr(model, "_orig_mod", model)
        if modality == "face_history":
            raw.fusion.train()
        elif modality == "face_only":
            if hasattr(raw, "classifier"):
                raw.classifier.train()
            else:
                raw.fc.train()
    total = batches = inputs_count = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        inputs, labels, _ = _prepare_batch(batch, modality, device, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(*inputs).squeeze(1)
            loss = criterion(logits, labels).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), base_config.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        batches += 1
        inputs_count += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss": total / max(batches, 1),
        "inputs": int(inputs_count),
        "seconds": elapsed,
        "throughput": inputs_count / max(elapsed, 1e-9),
        "memory_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }


def _clone(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _compile(model, modality, target, stage, enabled):
    if not enabled or not base_config.TORCH_COMPILE_ENABLED:
        return model, "eager"
    torch._dynamo.config.suppress_errors = True
    print(f"[compile] modality={modality} target={target} stage={stage}", flush=True)
    return (
        torch.compile(
            model, mode=base_config.TORCH_COMPILE_MODE,
            fullgraph=False, dynamic=False,
        ),
        f"torch_compile:{base_config.TORCH_COMPILE_MODE}",
    )


def _plot_history(history, path, modality, target):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (metric, title) in zip(axes.flat, (
        ("loss", "BCE loss"), ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC-AUC"), ("f1", "F1"),
    )):
        for stage, group in frame.groupby("stage", sort=False):
            axis.plot(group.global_epoch, group[f"train_{metric}"], label=f"{stage} train")
            axis.plot(group.global_epoch, group[f"val_{metric}"], "--", label=f"{stage} val")
        axis.set_title(title); axis.set_xlabel("Epoch"); axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    figure.suptitle(f"{modality} | {target} | true binary classification")
    figure.tight_layout(); figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _run_stage(
    stage, model, head, datasets, loaders, modality, target, device, criterion,
    learning_rate, epochs, patience_limit, history, run_dir, compile_enabled,
):
    _set_stage(model, head, modality, stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=base_config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=base_config.MIN_LEARNING_RATE
    )
    execution, backend = _compile(model, modality, target, stage, compile_enabled)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    best_state, best_score, patience, offset = None, -np.inf, 0, len(history)
    for epoch in range(1, epochs + 1):
        step = _train_epoch(
            execution, loaders["train_augmented"], modality, optimizer, scaler,
            criterion, device, stage == "head" and modality != "history_only",
        )
        train_metrics, _ = _evaluate(
            execution, loaders["train"], datasets["train"], modality,
            criterion, device, "train",
        )
        val_metrics, _ = _evaluate(
            execution, loaders["val"], datasets["val"], modality,
            criterion, device, "val",
        )
        score = val_metrics["balanced_accuracy"]
        if not np.isfinite(score):
            score = -val_metrics["loss"]
        row = {
            "modality": modality, "target": target, "stage": stage,
            "stage_epoch": epoch, "global_epoch": offset + epoch,
            "train_optimization_loss": step["loss"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_model_inputs": step["inputs"],
            "train_seconds": step["seconds"],
            "train_inputs_per_second": step["throughput"],
            "peak_gpu_memory_gb": step["memory_gb"],
            "execution_backend": backend,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if score > best_score + 1e-4:
            best_score, best_state, patience, marker = score, _clone(model), 0, "*"
        else:
            patience += 1; marker = ""
        print(
            f"[epoch] modality={modality} task={target} stage={stage} "
            f"{epoch:03d}/{epochs} train_loss={step['loss']:.4f} "
            f"train_bACC={train_metrics['balanced_accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bACC={val_metrics['balanced_accuracy']:.4f} "
            f"val_AUC={val_metrics['roc_auc']:.4f} val_F1={val_metrics['f1']:.4f} "
            f"throughput={step['throughput']:.1f}/s mem={step['memory_gb']:.2f}GiB "
            f"patience={patience}/{patience_limit}{marker}", flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] modality={modality} task={target} stage={stage}", flush=True)
            break
    del execution
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {modality}/{target}/{stage}")
    return best_state, best_score


def train_task(
    modality: str, target: str, device_id: int, seed: int,
    smoke: bool = False,
):
    if modality not in MODALITIES:
        raise ValueError(modality)
    seed_everything(seed)
    torch.cuda.set_device(device_id)
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{device_id}")
    output_dir = EXPERIMENT_DIRS[modality] / "outputs"
    run_dir = output_dir / "runs" / (
        Path("efficientnet_b0") / target if modality != "history_only" else Path(target)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    records, history_store = load_task(target)
    split_records = {
        split: records[records.split.eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    frame_index = None if modality == "history_only" else FrameOffsetIndex.load(FRAME_INDEX_PATH)
    datasets, loaders = {}, {}
    if modality == "history_only":
        datasets["train_augmented"], loaders["train_augmented"] = _history_loader(
            split_records["train"], history_store, True, seed
        )
        for split in ("train", "val", "test"):
            datasets[split], loaders[split] = _history_loader(
                split_records[split], history_store, False, seed
            )
    else:
        datasets["train_augmented"], loaders["train_augmented"] = _image_loader(
            modality, frame_index, split_records["train"], history_store, True
        )
        for split in ("train", "val", "test"):
            datasets[split], loaders[split] = _image_loader(
                modality, frame_index, split_records[split], history_store, False
            )
    negatives = int(split_records["train"].binary_label.eq(0).sum())
    positives = int(split_records["train"].binary_label.eq(1).sum())
    if min(negatives, positives) == 0:
        raise RuntimeError(f"Single-class training split for {target}")
    pos_weight = negatives / positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device), reduction="none"
    )
    model, head, weight_path = _build_model(modality)
    model = model.to(device)
    if modality != "history_only":
        model = model.to(memory_format=torch.channels_last)
    total_parameters = sum(p.numel() for p in model.parameters())
    train_inputs = (
        datasets["train_augmented"].model_input_count
        if modality != "history_only" else len(datasets["train_augmented"])
    )
    print(
        f"[job-start] modality={modality} task={target} device={device} "
        f"train/val/test videos={len(split_records['train'])}/"
        f"{len(split_records['val'])}/{len(split_records['test'])} "
        f"train_neg/pos={negatives}/{positives} pos_weight={pos_weight:.6f} "
        f"train_inputs={train_inputs} parameters={total_parameters}", flush=True,
    )
    history = []
    head_epochs = 1 if smoke else base_config.HEAD_MAX_EPOCHS
    fine_epochs = 1 if smoke else base_config.FINETUNE_MAX_EPOCHS
    head_patience = 1 if smoke else base_config.HEAD_PATIENCE
    fine_patience = 1 if smoke else base_config.FINETUNE_PATIENCE
    head_state, head_score = _run_stage(
        "head", model, head, datasets, loaders, modality, target, device,
        criterion, base_config.HEAD_LEARNING_RATE, head_epochs, head_patience,
        history, run_dir, not smoke,
    )
    model.load_state_dict(head_state)
    fine_state, fine_score = _run_stage(
        "finetune", model, head, datasets, loaders, modality, target, device,
        criterion, base_config.FINETUNE_LEARNING_RATE, fine_epochs, fine_patience,
        history, run_dir, not smoke,
    )
    selected_stage, state = (
        ("finetune", fine_state) if fine_score >= head_score else ("head", head_state)
    )
    model.load_state_dict(state)
    metric_rows, prediction_rows = [], []
    for split in ("train", "val", "test"):
        metrics, predictions = _evaluate(
            model, loaders[split], datasets[split], modality, criterion, device, split
        )
        metric_rows.append({
            "modality": modality, "architecture": (
                "history_only_head32" if modality == "history_only" else "efficientnet_b0"
            ),
            "target": target, "split": split, "selected_stage": selected_stage,
            "decision_threshold": 0.5, "pos_weight": pos_weight, **metrics,
        })
        prediction_rows.append(predictions)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    _plot_history(history, run_dir / "history.png", modality, target)
    torch.save({
        "schema_version": 1, "task_type": "binary_classification",
        "modality": modality, "target": target, "seed": seed,
        "selected_stage": selected_stage, "state_dict": state,
        "loss": "BCEWithLogitsLoss", "pos_weight": pos_weight,
        "decision_threshold": 0.5,
        "pretrained_weight_path": str(weight_path) if weight_path else None,
    }, run_dir / "model.pt")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1, "task_type": "true_binary_classification",
        "modality": modality, "target": target, "seed": seed,
        "reference_records": str(
            (PREPARED_DIR if target == "total_bilirubin_high" else REFERENCE_DIR)
            / "task_records" / f"{target}.csv"
        ),
        "split_policy": "exact reuse of the patient-disjoint regression split",
        "loss": "BCEWithLogitsLoss",
        "pos_weight": pos_weight, "decision_threshold": 0.5,
        "selected_stage": selected_stage,
        "head_hidden_features": base_config.HEAD_HIDDEN_FEATURES,
        "frames_per_video": base_config.FRAMES_PER_VIDEO,
        "training_views": list(base_config.VIEW_NAMES) if modality != "history_only" else [],
    }, indent=2), encoding="utf-8")
    test = metric_rows[-1]
    print(
        f"[job-complete] modality={modality} task={target} selected={selected_stage} "
        f"test_bACC={test['balanced_accuracy']:.4f} test_AUC={test['roc_auc']:.4f} "
        f"test_F1={test['f1']:.4f}", flush=True,
    )
    for dataset in datasets.values():
        if hasattr(dataset, "close"):
            dataset.close()
    return metric_rows
