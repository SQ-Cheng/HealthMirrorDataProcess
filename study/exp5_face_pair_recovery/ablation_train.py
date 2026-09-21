"""Train one-sided Exp5 ablations with the main experiment's frozen data."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from study.exp4.frame_index import FrameOffsetIndex

from .config import (
    CACHE_DIR, EVAL_BATCH_SIZE, EVAL_NUM_WORKERS,
    FINETUNE_BACKBONE_LEARNING_RATE, FINETUNE_HEAD_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, GRAD_CLIP_NORM,
    HEAD_LEARNING_RATE, HEAD_MAX_EPOCHS, HEAD_PATIENCE, MIN_LEARNING_RATE,
    OUTPUT_DIR, PREFETCH_FACTOR, SEED, TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZE, TRAIN_VIEWS, WEIGHT_DECAY,
    TARGET_COLUMN,
)
from .data import SingleFrameDataset
from .models import (
    build_ablation_model, freeze_backbone, head_parameters, last_stage_parameters,
    parameter_counts, train_head_modules, unfreeze_last_stage,
)
from .plot_results import plot_results
from .train import (
    _clone, _compile, _loss, _metrics, _plot_history, _prepare, seed_everything,
)


MODES = ("post_only", "pre_only")
GENERATED_FILES = (
    "history.csv", "metrics.csv", "model.pt", "run_manifest.json",
    "training_history.png", "video_predictions.csv",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loader(index, records, mode, train):
    dataset = SingleFrameDataset(
        index, records, mode, TRAIN_VIEWS if train else ("original",), train,
    )
    workers = TRAIN_NUM_WORKERS if train else EVAL_NUM_WORKERS
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_SOURCE_BATCH_SIZE if train else EVAL_BATCH_SIZE,
        shuffle=train, num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
    )
    return dataset, loader


def _train_epoch(model, loader, optimizer, scaler, device, frozen):
    model.eval() if frozen else model.train()
    raw = getattr(model, "_orig_mod", model)
    if frozen:
        train_head_modules(raw)
    total = weight_total = inputs = 0.0
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter()
    for images, target, _, codes, weights in loader:
        images = _prepare(images, codes, device)
        repeat = codes.shape[1]
        target = target.repeat_interleave(repeat).to(device, non_blocking=True)
        weights = weights.repeat_interleave(repeat).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(images).squeeze(1)
            loss = _loss(prediction, target, weights)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer); scaler.update()
        batch_weight = float(weights.sum().detach().cpu())
        total += float(loss.detach().cpu()) * batch_weight
        weight_total += batch_weight; inputs += len(target)
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
    return {
        "loss": total / max(weight_total, 1e-8), "inputs": int(inputs),
        "seconds": elapsed, "throughput": inputs / max(elapsed, 1e-8),
        "memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


@torch.no_grad()
def _frame_predictions(model, loader, device):
    model.eval(); truths, predictions, rows = [], [], []
    for images, target, frame_rows, codes, _ in loader:
        images = _prepare(images, codes, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(images).squeeze(1)
        truths.append(target.numpy()); predictions.append(output.float().cpu().numpy())
        rows.append(frame_rows.numpy())
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(rows)


def _evaluate(model, dataset, loader, device, split):
    truth, prediction, rows = _frame_predictions(model, loader, device)
    video_rows = dataset.frame_video_rows[rows]
    aggregate = pd.DataFrame({
        "video_row": video_rows, "y_true": truth, "y_pred": prediction,
    }).groupby("video_row", as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
        frame_count=("y_pred", "size"), frame_prediction_std=("y_pred", "std"),
    )
    info = dataset.records.iloc[aggregate.video_row.to_numpy(int)][[
        "hospital_id", "pre_video_id", "video_id", TARGET_COLUMN,
        "postoperative_progress",
    ]].reset_index(drop=True)
    result = pd.concat([info, aggregate.drop(columns="video_row")], axis=1)
    result.insert(0, "split", split)
    return _metrics(result.y_true, result.y_pred), result


def _stage(
    stage, model, datasets, loaders, device, run_dir, history,
    epochs, patience_limit, optimizer,
):
    execution, backend = _compile(model, stage)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=MIN_LEARNING_RATE)
    best_state, best_mae, patience, offset = None, np.inf, 0, len(history)
    for epoch in range(1, epochs + 1):
        step = _train_epoch(
            execution, loaders["train_augmented"], optimizer, scaler, device,
            stage == "head",
        )
        train_metrics, _ = _evaluate(
            execution, datasets["train"], loaders["train"], device, "train"
        )
        val_metrics, _ = _evaluate(
            execution, datasets["val"], loaders["val"], device, "val"
        )
        row = {
            "stage": stage, "stage_epoch": epoch, "global_epoch": offset + epoch,
            "train_optimization_loss": step["loss"],
            "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
            **{f"train_{key}": value for key, value in train_metrics.items() if key not in {"n", "loss"}},
            **{f"val_{key}": value for key, value in val_metrics.items() if key not in {"n", "loss"}},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_model_inputs": step["inputs"], "train_seconds": step["seconds"],
            "train_inputs_per_second": step["throughput"],
            "peak_gpu_memory_gb": step["memory_gb"], "execution_backend": backend,
        }
        history.append(row); pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_metrics["mae"] < best_mae - 1e-4:
            best_mae, best_state, patience, marker = val_metrics["mae"], _clone(model), 0, "*"
        else:
            patience += 1; marker = ""
        print(
            f"[epoch] stage={stage} {epoch:03d}/{epochs} "
            f"train_loss={step['loss']:.5f} train_MAE={train_metrics['mae']:.4f} "
            f"train_R2={train_metrics['r2']:.4f} val_MAE={val_metrics['mae']:.4f} "
            f"val_RMSE={val_metrics['rmse']:.4f} val_R2={val_metrics['r2']:.4f} "
            f"val_r={val_metrics['pearson_r']:.4f} throughput={step['throughput']:.1f}/s "
            f"mem={step['memory_gb']:.2f}GiB patience={patience}/{patience_limit}{marker}",
            flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] stage={stage}", flush=True); break
    del execution
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint in {stage}")
    return best_state, best_mae


def train_ablation(records, frame_index, mode, seed, device_id, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_FILES:
        (run_dir / name).unlink(missing_ok=True)
    seed_everything(seed); torch.cuda.set_device(device_id); torch.set_num_threads(4)
    device = torch.device(f"cuda:{device_id}")
    split_records = {
        split: records[records.split.eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    augmented, augmented_loader = _loader(
        frame_index, split_records["train"], mode, True
    )
    datasets, loaders = {}, {"train_augmented": augmented_loader}
    for split in ("train", "val", "test"):
        datasets[split], loaders[split] = _loader(
            frame_index, split_records[split], mode, False
        )
    model, weight_path = build_ablation_model(mode)
    model = model.to(device, memory_format=torch.channels_last)
    freeze_backbone(model); total, trainable = parameter_counts(model)
    print(
        f"[job-start] mode={mode} device={device} "
        f"videos={len(split_records['train'])}/{len(split_records['val'])}/{len(split_records['test'])} "
        f"patients={split_records['train'].hospital_id.nunique()}/"
        f"{split_records['val'].hospital_id.nunique()}/"
        f"{split_records['test'].hospital_id.nunique()} "
        f"frames={len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])} "
        f"train_inputs={augmented.model_input_count} parameters={total} head_trainable={trainable}",
        flush=True,
    )
    history = []
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=HEAD_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    print(f"[stage-start] stage=head lr={HEAD_LEARNING_RATE:.1e}", flush=True)
    head_state, head_mae = _stage(
        "head", model, datasets, loaders, device, run_dir, history,
        HEAD_MAX_EPOCHS, HEAD_PATIENCE, optimizer,
    )
    model.load_state_dict(head_state); unfreeze_last_stage(model)
    _, trainable = parameter_counts(model)
    backbone = last_stage_parameters(model)
    head = head_parameters(model)
    optimizer = AdamW([
        {"params": backbone, "lr": FINETUNE_BACKBONE_LEARNING_RATE},
        {"params": head, "lr": FINETUNE_HEAD_LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    print(
        f"[stage-start] stage=last_stage backbone_lr={FINETUNE_BACKBONE_LEARNING_RATE:.1e} "
        f"head_lr={FINETUNE_HEAD_LEARNING_RATE:.1e} trainable={trainable}", flush=True,
    )
    fine_state, fine_mae = _stage(
        "last_stage", model, datasets, loaders, device, run_dir, history,
        FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, optimizer,
    )
    selected, state = (
        ("last_stage", fine_state) if fine_mae <= head_mae else ("head", head_state)
    )
    model.load_state_dict(state); metric_rows, predictions = [], []
    for split in ("train", "val", "test"):
        metrics, frame = _evaluate(
            model, datasets[split], loaders[split], device, split
        )
        metric_rows.append({
            "mode": mode, "seed": seed, "selected_stage": selected,
            "split": split, **metrics,
        })
        predictions.append(frame)
    pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    torch.save({
        "schema_version": 1, "mode": mode, "seed": seed,
        "architecture": "capacity_matched_single_input_efficientnet_b0",
        "selected_stage": selected, "state_dict": state,
        "pretrained_weight_path": str(weight_path),
    }, run_dir / "model.pt")
    _plot_history(
        history, run_dir / "training_history.png",
        title=f"Exp5 {mode.replace('_', '-')} recovery training",
    )
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "schema_version": 1, "mode": mode, "seed": seed,
        "selected_stage": selected, "head_best_val_mae": head_mae,
        "last_stage_best_val_mae": fine_mae,
        "missing_branch": "zero 64-dimensional embedding; encoder not executed",
        "fusion": "same 256-to-32 head as paired main experiment",
        "target": "equal-weight postoperative trajectory-deviation score",
        "train_views": list(TRAIN_VIEWS),
        "records_sha256": _sha256(run_dir / "records.csv"),
        "frame_index_sha256": _sha256(CACHE_DIR / "frame_offsets.npz"),
    }, indent=2), encoding="utf-8")
    test = metric_rows[-1]
    print(
        f"[job-complete] mode={mode} selected={selected} "
        f"test_MAE={test['mae']:.4f} test_R2={test['r2']:.4f} "
        f"test_r={test['pearson_r']:.4f}", flush=True,
    )


def smoke_test(records, frame_index, mode, device_id):
    device = torch.device(f"cuda:{device_id}"); torch.cuda.set_device(device_id)
    sample = records[records.split.eq("train")].head(4).reset_index(drop=True)
    dataset = SingleFrameDataset(frame_index, sample, mode)
    batch = [dataset[index] for index in range(4)]
    images, targets, _, codes, weights = zip(*batch)
    images = _prepare(torch.stack(images), torch.stack(codes), device)
    targets = torch.stack(targets).to(device); weights = torch.stack(weights).to(device)
    model, _ = build_ablation_model(mode); freeze_backbone(model); model.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(images).squeeze(1)
        loss = _loss(prediction, targets, weights)
    loss.backward(); total, trainable = parameter_counts(model)
    if not torch.isfinite(loss) or not prediction.detach().ge(0).all():
        raise RuntimeError("Invalid ablation smoke output")
    print(
        f"[smoke-ok] mode={mode} shape={tuple(images.shape)} loss={float(loss):.6f} "
        f"parameters={total} trainable={trainable}", flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_dir = OUTPUT_DIR / "ablations" / args.mode
    records_path = run_dir / "records.csv"
    frame_index_path = CACHE_DIR / "frame_offsets.npz"
    records = pd.read_csv(records_path, dtype={"hospital_id": str})
    frame_index = FrameOffsetIndex.load(frame_index_path)
    if records.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError("Main Exp5 records contain patient leakage")
    if args.smoke:
        smoke_test(records, frame_index, args.mode, args.device); return
    train_ablation(records, frame_index, args.mode, args.seed, args.device, run_dir)
    plot_results(run_dir)


if __name__ == "__main__":
    main()
