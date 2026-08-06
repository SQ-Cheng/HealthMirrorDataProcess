"""Train one history-only abnormal-score regressor for one laboratory task."""

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
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_LEARNING_RATE,
    MODEL_NAME,
    PATIENCE,
    SMOOTH_L1_BETA,
    WEIGHT_DECAY,
)
from .data import HistoryOnlyDataset
from .models import HistoryOnlyRegressor, parameter_counts


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    valid = np.isfinite(targets) & np.isfinite(predictions)
    targets, predictions = targets[valid], predictions[valid]
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
        "sign_threshold": 0.0,
    }
    if not len(targets):
        return result
    result.update(
        {
            "mae": float(mean_absolute_error(targets, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
            "median_ae": float(median_absolute_error(targets, predictions)),
            "r2": (
                float(r2_score(targets, predictions))
                if len(targets) > 1 and np.var(targets) > 0
                else np.nan
            ),
            "pearson_r": (
                float(np.corrcoef(targets, predictions)[0, 1])
                if len(targets) > 1
                and np.std(targets) > 0
                and np.std(predictions) > 0
                else np.nan
            ),
            "spearman_r": (
                float(
                    pd.Series(targets).rank().corr(
                        pd.Series(predictions).rank(), method="pearson"
                    )
                )
                if len(targets) > 1
                else np.nan
            ),
        }
    )
    non_boundary = ~np.isclose(targets, 0.0, atol=1e-12)
    sign_targets = (targets[non_boundary] > 0).astype(np.uint8)
    sign_predictions = (predictions[non_boundary] > 0).astype(np.uint8)
    result["sign_n"] = int(len(sign_targets))
    if len(sign_targets) and len(np.unique(sign_targets)) == 2:
        tn, fp, fn, tp = confusion_matrix(
            sign_targets, sign_predictions, labels=[0, 1]
        ).ravel()
        result.update(
            {
                "sign_accuracy": float(accuracy_score(sign_targets, sign_predictions)),
                "sign_balanced_accuracy": float(
                    balanced_accuracy_score(sign_targets, sign_predictions)
                ),
                "sign_f1": float(
                    f1_score(sign_targets, sign_predictions, zero_division=0)
                ),
                "sign_roc_auc": float(
                    roc_auc_score(sign_targets, predictions[non_boundary])
                ),
                "sign_average_precision": float(
                    average_precision_score(sign_targets, predictions[non_boundary])
                ),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
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


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    targets, predictions, row_indices = [], [], []
    with torch.no_grad():
        for history, mask, labels, rows in loader:
            history = history.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(history, mask)
                loss = criterion(output, labels).mean()
            total_loss += float(loss.cpu()) * len(labels)
            total_n += len(labels)
            targets.append(labels[:, 0].cpu().numpy())
            predictions.append(output[:, 0].float().cpu().numpy())
            row_indices.append(rows.numpy())
    return {
        "loss": total_loss / max(total_n, 1),
        "targets": np.concatenate(targets),
        "predictions": np.concatenate(predictions),
        "row_indices": np.concatenate(row_indices),
    }

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
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(frame.epoch, frame.train_eval_loss, label="train")
    axes[0].plot(frame.epoch, frame.val_loss, label="val")
    axes[0].set_title("SmoothL1 loss")
    axes[1].plot(frame.epoch, frame.train_mae, label="train MAE")
    axes[1].plot(frame.epoch, frame.val_mae, label="val MAE")
    axes[1].set_title("MAE")
    axes[2].plot(frame.epoch, frame.train_pearson_r, label="train Pearson r")
    axes[2].plot(frame.epoch, frame.val_pearson_r, label="val Pearson r")
    axes[2].set_title("Correlation")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"History-only Head32 | {target}")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train_task(target, records, history_store, run_dir, device, seed, max_epochs=None):
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
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = int(max_epochs or MAX_EPOCHS)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=1024.0
    )
    print(
        f"[job-start] model={MODEL_NAME} task={target} device={device} "
        f"train/val/test videos={len(datasets['train'])}/{len(datasets['val'])}/"
        f"{len(datasets['test'])} parameters={counts['total']} "
        f"history_encoder={counts['history_encoder']} head={counts['head']}",
        flush=True,
    )
    history_rows = []
    best_mae, best_state, patience = np.inf, None, 0
    for epoch in range(1, epochs + 1):
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
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu()) * len(labels)
            train_n += len(labels)
            optimizer_steps += 1
        train_seconds = time.perf_counter() - started
        train_eval = _evaluate(model, loaders["train"], criterion, device)
        val_eval = _evaluate(model, loaders["val"], criterion, device)
        train_metrics = regression_metrics(
            train_eval["targets"], train_eval["predictions"]
        )
        val_metrics = regression_metrics(val_eval["targets"], val_eval["predictions"])
        row = {
            "model": MODEL_NAME,
            "target": target,
            "stage": "history_only",
            "epoch": epoch,
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
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "optimizer_steps": optimizer_steps,
            "train_seconds": train_seconds,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)
        improved = val_metrics["mae"] < best_mae - 1e-4
        if improved:
            best_mae = val_metrics["mae"]
            best_state = _clone_state(model)
            patience, marker = 0, "*"
        else:
            patience, marker = patience + 1, ""
        print(
            f"[epoch] task={target} {epoch:03d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} train_MAE={row['train_mae']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_MAE={row['val_mae']:.4f} "
            f"val_r={row['val_pearson_r']:.4f} "
            f"val_sign_AUC={row['val_sign_auc']:.4f} "
            f"val_sign_bACC={row['val_sign_bacc']:.4f} "
            f"lr={row['learning_rate']:.2e} patience={patience}/{PATIENCE}{marker}",
            flush=True,
        )
        scheduler.step()
        if patience >= PATIENCE:
            print(f"[early-stop] task={target}", flush=True)
            break
    if best_state is None:
        raise RuntimeError(f"No finite checkpoint for {target}")
    model.load_state_dict(best_state, strict=True)
    metric_rows, prediction_rows = [], []
    for split in ("train", "val", "test"):
        evaluation = _evaluate(model, loaders[split], criterion, device)
        metrics = regression_metrics(evaluation["targets"], evaluation["predictions"])
        metric_rows.append(
            {"model": MODEL_NAME, "target": target, "split": split, **metrics}
        )
        dataset = datasets[split]
        selected = dataset.records.iloc[evaluation["row_indices"]].reset_index(drop=True)
        selected = selected[
            [
                "hospital_id",
                "video_id",
                "binary_label",
                "raw_value",
                "score_threshold",
                "abnormal_score",
            ]
        ].copy()
        selected.insert(0, "split", split)
        selected["y_true"] = evaluation["targets"]
        selected["y_pred"] = evaluation["predictions"]
        selected["residual"] = selected["y_pred"] - selected["y_true"]
        selected["history_count"] = dataset.history_count[evaluation["row_indices"]]
        selected["input_count"] = 1
        selected["model"] = MODEL_NAME
        selected["target"] = target
        prediction_rows.append(selected)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        run_dir / "video_predictions.csv", index=False
    )
    _plot_history(history_rows, run_dir / "history.png", target)
    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model": MODEL_NAME,
            "target": target,
            "task_type": "abnormal_score_regression",
            "input": "prior_same-analyte_history_only",
            "parameters": counts,
            "head_hidden_features": 32,
            "selected_epoch": int(
                pd.DataFrame(history_rows).sort_values("val_mae").iloc[0]["epoch"]
            ),
            "loss": "unweighted SmoothL1",
            "smooth_l1_beta": SMOOTH_L1_BETA,
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    verifier = HistoryOnlyRegressor()
    verifier.load_state_dict(loaded["model_state_dict"], strict=True)
    if loaded["target"] != target or loaded["model"] != MODEL_NAME:
        raise RuntimeError(f"Saved checkpoint metadata mismatch for {target}")
    test = metrics_frame.loc[metrics_frame["split"].eq("test")].iloc[0]
    print(
        f"[job-done] task={target} test_MAE={test.mae:.4f} "
        f"test_RMSE={test.rmse:.4f} test_r={test.pearson_r:.4f} "
        f"test_sign_AUC={test.sign_roc_auc:.4f} "
        f"checkpoint_sha256={_sha256(checkpoint_path)}",
        flush=True,
    )
    return metrics_frame, {
        "target": target,
        "status": "ok",
        "run_dir": str(run_dir.resolve()),
        "model_pt_bytes": os.path.getsize(checkpoint_path),
        "model_pt_sha256": _sha256(checkpoint_path),
        **counts,
    }
