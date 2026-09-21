"""Train one leakage-safe history-only raw-value regressor per lab target."""

import hashlib
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    GRAD_CLIP_NORM,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    MIN_LEARNING_RATE,
    MODEL_NAME,
    SCORE_DEFINITIONS,
    SMOOTH_L1_BETA,
    WEIGHT_DECAY,
)
from .data import HistoryOnlyDataset
from .models import HistoryOnlyRegressor, parameter_counts


def regression_metrics(targets, predictions, thresholds, direction):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    valid = np.isfinite(targets) & np.isfinite(predictions) & np.isfinite(thresholds)
    targets, predictions, thresholds = (
        targets[valid], predictions[valid], thresholds[valid]
    )
    result = {
        "n": int(len(targets)),
        "mae": np.nan,
        "rmse": np.nan,
        "median_ae": np.nan,
        "r2": np.nan,
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "sign_n": 0,
        "sign_accuracy": np.nan,
        "sign_balanced_accuracy": np.nan,
        "sign_f1": np.nan,
        "sign_roc_auc": np.nan,
        "sign_average_precision": np.nan,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "sign_threshold": np.nan,
        "sign_threshold_policy": f"per-video clinical {direction} threshold",
    }
    if not len(targets):
        return result
    result.update({
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "median_ae": float(median_absolute_error(targets, predictions)),
        "r2": (
            float(r2_score(targets, predictions))
            if len(targets) > 1 and np.var(targets) > 0 else np.nan
        ),
        "pearson_r": (
            float(np.corrcoef(targets, predictions)[0, 1])
            if len(targets) > 1
            and np.std(targets) > 0
            and np.std(predictions) > 0
            else np.nan
        ),
        "spearman_r": (
            float(pd.Series(targets).rank().corr(pd.Series(predictions).rank()))
            if len(targets) > 1 else np.nan
        ),
    })
    non_boundary = ~np.isclose(targets, thresholds, atol=1e-12)
    if direction == "low":
        sign_targets = (
            targets[non_boundary] < thresholds[non_boundary]
        ).astype(np.uint8)
        sign_predictions = (
            predictions[non_boundary] < thresholds[non_boundary]
        ).astype(np.uint8)
        decision = thresholds[non_boundary] - predictions[non_boundary]
    elif direction == "high":
        sign_targets = (
            targets[non_boundary] > thresholds[non_boundary]
        ).astype(np.uint8)
        sign_predictions = (
            predictions[non_boundary] > thresholds[non_boundary]
        ).astype(np.uint8)
        decision = predictions[non_boundary] - thresholds[non_boundary]
    else:
        raise ValueError(f"Unsupported score direction: {direction}")
    result["sign_n"] = int(len(sign_targets))
    if len(sign_targets) and len(np.unique(sign_targets)) == 2:
        tn, fp, fn, tp = confusion_matrix(
            sign_targets, sign_predictions, labels=[0, 1]
        ).ravel()
        result.update({
            "sign_accuracy": float(accuracy_score(sign_targets, sign_predictions)),
            "sign_balanced_accuracy": float(
                balanced_accuracy_score(sign_targets, sign_predictions)
            ),
            "sign_f1": float(f1_score(sign_targets, sign_predictions, zero_division=0)),
            "sign_roc_auc": float(roc_auc_score(sign_targets, decision)),
            "sign_average_precision": float(
                average_precision_score(sign_targets, decision)
            ),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return result


def _loader(dataset, shuffle, seed):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    targets, predictions, row_indices = [], [], []
    for history, mask, labels, rows in loader:
        history = history.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(history, mask)
            loss = criterion(output, labels_device).mean()
        total_loss += float(loss.cpu()) * len(labels)
        total_n += len(labels)
        targets.append(labels.numpy())
        predictions.append(output[:, 0].float().cpu().numpy())
        row_indices.append(rows.numpy())
    return {
        "loss": total_loss / max(total_n, 1),
        "targets": np.concatenate(targets),
        "predictions": np.concatenate(predictions),
        "row_indices": np.concatenate(row_indices),
    }


def _raw_evaluation(evaluation, dataset, target, target_scaler):
    rows = evaluation["row_indices"].astype(np.int64)
    records = dataset.records.iloc[rows].reset_index(drop=True)
    y_true_scaled = evaluation["targets"].astype(np.float32)
    y_pred_scaled = evaluation["predictions"].astype(np.float64)
    y_true = records["raw_value"].to_numpy(np.float64)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    expected_scaled = target_scaler.transform(y_true).astype(np.float32)
    if not np.array_equal(y_true_scaled, expected_scaled):
        mismatches = int(np.count_nonzero(y_true_scaled != expected_scaled))
        raise AssertionError(
            "History-only scaled labels are misaligned with raw values: "
            f"{mismatches}/{len(y_true_scaled)}"
        )
    metrics = regression_metrics(
        y_true,
        y_pred,
        records["score_threshold"],
        SCORE_DEFINITIONS[target]["direction"],
    )
    return metrics, records, y_true_scaled, y_pred_scaled, y_true, y_pred


def _clone_state(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot_history(history, path, target):
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for stage, group in frame.groupby("stage", sort=False):
        axes[0].plot(group.global_epoch, group.train_loss, label=f"{stage} train")
        axes[0].plot(
            group.global_epoch, group.val_loss, linestyle="--", label=f"{stage} val"
        )
        axes[1].plot(group.global_epoch, group.train_mae, label=f"{stage} train")
        axes[1].plot(
            group.global_epoch, group.val_mae, linestyle="--", label=f"{stage} val"
        )
        axes[2].plot(
            group.global_epoch, group.val_pearson_r, label=f"{stage} val r"
        )
    axes[0].set_title("SmoothL1 loss")
    axes[1].set_title("Raw-value MAE")
    axes[2].set_title("Validation Pearson r")
    for axis in axes:
        axis.set_xlabel("Global epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[2].set_ylim(-1.05, 1.05)
    figure.suptitle(f"History-only Head32 | {target}")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _run_stage(
    stage,
    model,
    loaders,
    datasets,
    criterion,
    device,
    target_scaler,
    learning_rate,
    max_epochs,
    patience_limit,
    history_rows,
    run_dir,
    target,
):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(max_epochs, 1), eta_min=MIN_LEARNING_RATE
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=1024.0
    )
    best_mae, best_state, patience = np.inf, None, 0
    start_global_epoch = len(history_rows)
    for stage_epoch in range(1, max_epochs + 1):
        model.train()
        started = time.perf_counter()
        train_loss, train_n, optimizer_steps = 0.0, 0, 0
        for history, mask, labels, _ in loaders["train_shuffle"]:
            history = history.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(history, mask)
                loss = criterion(output, labels).mean()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += float(loss.detach().cpu()) * len(labels)
            train_n += len(labels)
            optimizer_steps += 1
        train_seconds = time.perf_counter() - started
        train_eval = _evaluate(model, loaders["train"], criterion, device)
        val_eval = _evaluate(model, loaders["val"], criterion, device)
        train_metrics = _raw_evaluation(
            train_eval, datasets["train"], target, target_scaler
        )[0]
        val_metrics = _raw_evaluation(
            val_eval, datasets["val"], target, target_scaler
        )[0]
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "architecture": MODEL_NAME,
            "target": target,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "global_epoch": start_global_epoch + stage_epoch,
            "train_loss": train_loss / max(train_n, 1),
            "train_eval_loss": train_eval["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "train_pearson_r": train_metrics["pearson_r"],
            "train_spearman_r": train_metrics["spearman_r"],
            "train_sign_bacc": train_metrics["sign_balanced_accuracy"],
            "train_sign_auc": train_metrics["sign_roc_auc"],
            "val_loss": val_eval["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_spearman_r": val_metrics["spearman_r"],
            "val_sign_bacc": val_metrics["sign_balanced_accuracy"],
            "val_sign_auc": val_metrics["sign_roc_auc"],
            "learning_rate": current_lr,
            "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
        improved = val_metrics["mae"] < best_mae - 1e-4
        if improved:
            best_mae, best_state, patience, marker = (
                val_metrics["mae"], _clone_state(model), 0, "*"
            )
        else:
            patience, marker = patience + 1, ""
        print(
            f"[epoch] task={target} stage={stage} {stage_epoch:03d}/{max_epochs} "
            f"train_loss={row['train_loss']:.4f} train_MAE={row['train_mae']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_MAE={row['val_mae']:.4f} "
            f"val_r={row['val_pearson_r']:.4f} "
            f"val_sign_AUC={row['val_sign_auc']:.4f} "
            f"val_sign_bACC={row['val_sign_bacc']:.4f} "
            f"lr={current_lr:.2e} patience={patience}/{patience_limit}{marker}",
            flush=True,
        )
        scheduler.step()
        if patience >= patience_limit:
            print(f"[early-stop] task={target} stage={stage}", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint for {target}/{stage}")
    return best_state, best_mae


def train_task(
    target,
    records,
    history_store,
    target_scaler,
    run_dir,
    device,
    seed,
    head_epochs=HEAD_MAX_EPOCHS,
    finetune_epochs=FINETUNE_MAX_EPOCHS,
    head_patience=HEAD_PATIENCE,
    finetune_patience=FINETUNE_PATIENCE,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        split: records.loc[records["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    datasets = {
        split: HistoryOnlyDataset(split_records[split], history_store)
        for split in split_records
    }
    loaders = {
        "train_shuffle": _loader(datasets["train"], True, seed),
        **{
            split: _loader(datasets[split], False, seed)
            for split in ("train", "val", "test")
        },
    }
    model = HistoryOnlyRegressor().to(device)
    counts = parameter_counts(model)
    if any("backbone" in name or "image" in name for name, _ in model.named_parameters()):
        raise AssertionError("Image/backbone parameters survived in history-only model")
    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA, reduction="none")
    print(
        f"[job-start] model={MODEL_NAME} task={target} device={device} "
        f"train/val/test videos={len(datasets['train'])}/{len(datasets['val'])}/"
        f"{len(datasets['test'])} parameters={counts['total']} "
        f"history_encoder={counts['history_encoder']} head={counts['head']}",
        flush=True,
    )
    history_rows = []
    head_state, head_mae = _run_stage(
        "head", model, loaders, datasets, criterion, device, target_scaler,
        HEAD_LEARNING_RATE, int(head_epochs), int(head_patience), history_rows,
        run_dir, target,
    )
    torch.save(head_state, run_dir / "stage_head_best.pt")
    model.load_state_dict(head_state, strict=True)
    finetune_state, finetune_mae = _run_stage(
        "finetune", model, loaders, datasets, criterion, device, target_scaler,
        FINETUNE_LEARNING_RATE, int(finetune_epochs), int(finetune_patience),
        history_rows, run_dir, target,
    )
    torch.save(finetune_state, run_dir / "stage_finetune_best.pt")
    selected_stage = "finetune" if finetune_mae <= head_mae else "head"
    best_state = finetune_state if selected_stage == "finetune" else head_state
    model.load_state_dict(best_state, strict=True)

    metric_rows, prediction_rows = [], []
    for split in ("train", "val", "test"):
        evaluation = _evaluate(model, loaders[split], criterion, device)
        metrics, selected, y_true_scaled, y_pred_scaled, y_true, y_pred = (
            _raw_evaluation(evaluation, datasets[split], target, target_scaler)
        )
        metric_rows.append({
            "architecture": MODEL_NAME,
            "target": target,
            "split": split,
            "selected_stage": selected_stage,
            **metrics,
        })
        predictions = selected[[
            "hospital_id", "video_id", "binary_label", "raw_value",
            "score_threshold", "score_scale",
        ]].copy()
        predictions.insert(0, "split", split)
        predictions["y_true_scaled"] = y_true_scaled
        predictions["y_pred_scaled"] = y_pred_scaled
        predictions["y_true"] = y_true
        predictions["y_pred"] = y_pred
        predictions["residual"] = y_pred - y_true
        predictions["history_count"] = datasets[split].history_count[
            evaluation["row_indices"]
        ]
        predictions["input_count"] = 1
        predictions["architecture"] = MODEL_NAME
        predictions["target"] = target
        prediction_rows.append(predictions)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    _plot_history(history_rows, run_dir / "history.png", target)
    checkpoint_path = run_dir / "model.pt"
    torch.save({
        "model_state_dict": best_state,
        "architecture": MODEL_NAME,
        "target": target,
        "task_type": "robust_scaled_raw_value_regression",
        "target_scaler": target_scaler.to_dict(),
        "input": "prior_same-analyte_history_only",
        "parameters": counts,
        "head_hidden_features": 32,
        "selected_stage": selected_stage,
        "loss": "unweighted SmoothL1",
        "smooth_l1_beta": SMOOTH_L1_BETA,
    }, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    verifier = HistoryOnlyRegressor()
    verifier.load_state_dict(loaded["model_state_dict"], strict=True)
    if loaded["target"] != target or loaded["architecture"] != MODEL_NAME:
        raise RuntimeError(f"Saved checkpoint metadata mismatch for {target}")
    test = metrics_frame.loc[metrics_frame["split"].eq("test")].iloc[0]
    print(
        f"[job-done] task={target} selected={selected_stage} "
        f"test_MAE={test.mae:.4f} test_RMSE={test.rmse:.4f} "
        f"test_r={test.pearson_r:.4f} test_sign_AUC={test.sign_roc_auc:.4f} "
        f"checkpoint_sha256={_sha256(checkpoint_path)}",
        flush=True,
    )
    return metrics_frame, {
        "target": target,
        "status": "ok",
        "run_dir": str(run_dir.resolve()),
        "model_pt_bytes": os.path.getsize(checkpoint_path),
        "model_pt_sha256": _sha256(checkpoint_path),
        "selected_stage": selected_stage,
        **counts,
    }
