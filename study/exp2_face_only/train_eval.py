"""Train one single-frame RGB model per task using 20 non-adjacent frame samples."""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .config import (
    AUGMENT_BRIGHTNESS,
    AUGMENT_CONTRAST,
    AUGMENT_CROP_MIN_SCALE,
    AUGMENT_HORIZONTAL_FLIP,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LOG_DIR,
    MAX_EPOCHS,
    MIN_LEARNING_RATE,
    MIN_TRAIN_PATIENTS_PER_CLASS,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    NUM_WORKERS,
    OUTPUT_DIR,
    POS_WEIGHT_MAX,
    SEED,
    TARGETS,
    WEIGHT_DECAY,
)
from .models import SingleFrameRGBNet, count_parameters


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _channel_statistics(face):
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, len(face), 32):
        batch = face[start:start + 32].astype(np.float64) / 255.0
        total += batch.sum(axis=(0, 1, 3, 4))
        total_sq += np.square(batch).sum(axis=(0, 1, 3, 4))
        count += batch.shape[0] * batch.shape[1] * batch.shape[3] * batch.shape[4]
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class SingleFrameTaskDataset(Dataset):
    def __init__(self, face, video_indices, frame_indices, labels, row_indices, mean, std, augment=False):
        self.face = face
        self.video_indices = video_indices
        self.frame_indices = frame_indices.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.row_indices = row_indices.astype(np.int64)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        face = torch.from_numpy(
            self.face[self.video_indices[index], self.frame_indices[index]]
        ).float().div_(255.0)
        if self.augment:
            if AUGMENT_HORIZONTAL_FLIP and np.random.rand() < 0.5:
                face = torch.flip(face, dims=(-1,))
            if AUGMENT_CROP_MIN_SCALE < 1.0:
                scale = np.random.uniform(AUGMENT_CROP_MIN_SCALE, 1.0)
                height, width = face.shape[-2:]
                crop_h = max(8, int(round(height * scale)))
                crop_w = max(8, int(round(width * scale)))
                top = np.random.randint(0, height - crop_h + 1)
                left = np.random.randint(0, width - crop_w + 1)
                face = functional.interpolate(
                    face[:, top:top + crop_h, left:left + crop_w].unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            if AUGMENT_CONTRAST > 0:
                factor = 1.0 + np.random.uniform(-AUGMENT_CONTRAST, AUGMENT_CONTRAST)
                spatial_mean = face.mean(dim=(-2, -1), keepdim=True)
                face = (face - spatial_mean) * factor + spatial_mean
            if AUGMENT_BRIGHTNESS > 0:
                face = face * (
                    1.0 + np.random.uniform(-AUGMENT_BRIGHTNESS, AUGMENT_BRIGHTNESS)
                )
            face = face.clamp_(0.0, 1.0)
        face = (face - self.mean) / self.std
        return (
            face,
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(self.row_indices[index], dtype=torch.long),
        )

def _patient_level_split_for_task(manifest, target, seed):
    target_values = pd.to_numeric(manifest[target], errors="coerce")
    valid = manifest.loc[target_values.notna(), ["hospital_id", target]].copy()
    valid[target] = pd.to_numeric(valid[target], errors="coerce")
    patient_labels = valid.groupby("hospital_id")[target].max().reset_index()
    patient_labels["hospital_id"] = patient_labels["hospital_id"].astype(str)

    if len(patient_labels) < 5:
        return None
    class_counts = patient_labels[target].value_counts()
    if patient_labels[target].nunique() < 2 or class_counts.min() < 3:
        return None
    try:
        first_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.40, random_state=seed
        )
        train_index, temporary_index = next(
            first_split.split(patient_labels["hospital_id"], patient_labels[target])
        )
        temporary = patient_labels.iloc[temporary_index].reset_index(drop=True)
        second_split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed + 1
        )
        validation_index, test_index = next(
            second_split.split(temporary["hospital_id"], temporary[target])
        )
    except ValueError:
        return None
    return {
        "train": set(patient_labels.iloc[train_index]["hospital_id"]),
        "val": set(temporary.iloc[validation_index]["hospital_id"]),
        "test": set(temporary.iloc[test_index]["hospital_id"]),
    }


def _task_split_data(manifest, target, split):
    labels = pd.to_numeric(manifest[target], errors="coerce")
    hospital_ids = manifest["hospital_id"].astype(str).to_numpy()
    video_indices = pd.to_numeric(manifest["video_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    frame_indices = pd.to_numeric(manifest["frame_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    result = {}
    for split_name in ("train", "val", "test"):
        mask = labels.notna().to_numpy() & np.asarray(
            [hospital_id in split[split_name] for hospital_id in hospital_ids]
        )
        row_indices = np.flatnonzero(mask)
        result[split_name] = {
            "row_indices": row_indices,
            "video_indices": video_indices[row_indices],
            "frame_indices": frame_indices[row_indices],
            "labels": labels.iloc[row_indices].to_numpy(dtype=np.float32),
        }
    return result


def _class_patient_counts(manifest, row_indices, labels):
    task_rows = manifest.iloc[row_indices][["hospital_id"]].copy()
    task_rows["label"] = labels
    grouped = task_rows.groupby("hospital_id")["label"]
    positive_patients = int(grouped.max().gt(0.5).sum())
    negative_patients = int(grouped.min().lt(0.5).sum())
    return positive_patients, negative_patients


def _make_loader(face, data, mean, std, shuffle, augment):
    dataset = SingleFrameTaskDataset(
        face=face,
        video_indices=data["video_indices"],
        frame_indices=data["frame_indices"],
        labels=data["labels"],
        row_indices=data["row_indices"],
        mean=mean,
        std=std,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def _optimal_threshold(y_true, y_score):
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    index = int(np.argmax((true_positive_rate - false_positive_rate)[finite]))
    return float(thresholds[finite][index])


def _binary_metrics(y_true, y_score, threshold=0.5):
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid].astype(int)
    y_score = y_score[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "n": int(len(y_true)),
            "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
            "threshold": float(threshold),
        }
    prediction = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
    }


def _train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    batches = 0
    use_amp = device.type == "cuda"
    for faces, labels, _ in loader:
        faces = faces.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(faces)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    probabilities = []
    labels_out = []
    row_indices = []
    use_amp = device.type == "cuda"
    for faces, labels, rows in loader:
        faces = faces.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True).unsqueeze(1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(faces)
            loss = criterion(logits, labels_device)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
        labels_out.append(labels.numpy())
        row_indices.append(rows.numpy())
        losses.append(float(loss.cpu()))
    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels_out),
        "row_indices": np.concatenate(row_indices),
    }


def _plot_loss_curves(loss_history, output_dir):
    tasks = loss_history["target"].unique()
    if len(tasks) == 0:
        return
    n_columns = min(3, len(tasks))
    n_rows = int(np.ceil(len(tasks) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(6 * n_columns, 4.5 * n_rows), squeeze=False
    )
    axes = axes.flatten()
    for index, target in enumerate(tasks):
        axis = axes[index]
        history = loss_history[loss_history["target"] == target]
        epochs = history["epoch"].to_numpy()
        axis.plot(epochs, history["train_loss"], "b-", label="Train loss", alpha=0.75)
        axis.plot(epochs, history["val_loss"], "r-", label="Val loss", alpha=0.75)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        second_axis = axis.twinx()
        second_axis.plot(epochs, history["val_bacc"], "g-", label="Val bACC")
        second_axis.plot(epochs, history["val_roc_auc"], "m-", label="Val ROC-AUC")
        second_axis.set_ylim(-0.05, 1.05)
        second_axis.set_ylabel("Score")
        lines_a, labels_a = axis.get_legend_handles_labels()
        lines_b, labels_b = second_axis.get_legend_handles_labels()
        axis.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc="best")
        axis.set_title(target, fontsize=9)
    for index in range(len(tasks), len(axes)):
        axes[index].set_visible(False)
    figure.suptitle("Exp2 Aug20 Non-Adjacent RGB: Per-Task Models", fontsize=14, y=1.01)
    figure.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Loss and metric curves saved to {path}", flush=True)


def _train_one_task(face, manifest, target, split, mean, std, device):
    data = _task_split_data(manifest, target, split)
    train_labels = data["train"]["labels"]
    n_positive = int((train_labels > 0.5).sum())
    n_negative = int((train_labels < 0.5).sum())
    positive_patients, negative_patients = _class_patient_counts(
        manifest, data["train"]["row_indices"], train_labels
    )
    if (
        n_positive < MIN_TRAIN_SAMPLES_PER_CLASS
        or n_negative < MIN_TRAIN_SAMPLES_PER_CLASS
        or positive_patients < MIN_TRAIN_PATIENTS_PER_CLASS
        or negative_patients < MIN_TRAIN_PATIENTS_PER_CLASS
    ):
        return None, {
            "target": target,
            "split": "test",
            "status": "skipped",
            "reason": (
                f"insufficient train class: pos={n_positive}/{positive_patients} patients, "
                f"neg={n_negative}/{negative_patients} patients"
            ),
        }, [], [], []

    if any(len(data[name]["labels"]) == 0 for name in ("train", "val", "test")):
        return None, {
            "target": target,
            "split": "test",
            "status": "skipped",
            "reason": "empty patient split",
        }, [], [], []

    loaders = {
        "train": _make_loader(face, data["train"], mean, std, shuffle=True, augment=True),
        "val": _make_loader(face, data["val"], mean, std, shuffle=False, augment=False),
        "test": _make_loader(face, data["test"], mean, std, shuffle=False, augment=False),
    }
    pos_weight_value = float(
        min(POS_WEIGHT_MAX, max(1.0, n_negative / max(n_positive, 1)))
    )
    model = SingleFrameRGBNet().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(
        f"  Samples: train={len(train_labels)} val={len(data['val']['labels'])} "
        f"test={len(data['test']['labels'])}; "
        f"pos={n_positive}/{positive_patients} patients; "
        f"neg={n_negative}/{negative_patients} patients; "
        f"pos_weight={pos_weight_value:.2f}",
        flush=True,
    )

    best_score = -np.inf
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _train_epoch(model, loaders["train"], optimizer, scaler, criterion, device)
        validation = _evaluate(model, loaders["val"], criterion, device)
        validation_metrics = _binary_metrics(
            validation["labels"], validation["probabilities"], threshold=0.5
        )
        validation_auc = validation_metrics["roc_auc"]
        validation_bacc = validation_metrics["balanced_accuracy"]
        score = validation_auc if np.isfinite(validation_auc) else -validation["loss"]
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "target": target,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_bacc": validation_bacc,
            "val_roc_auc": validation_auc,
            "learning_rate": learning_rate,
        })
        print(
            f"  {target} epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_AUC={validation_auc:.4f} "
            f"val_bACC={validation_bacc:.4f}",
            flush=True,
        )
        if score > best_score + 1e-4:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        scheduler.step()
        if patience >= EARLY_STOPPING_PATIENCE:
            print(f"  {target}: early stopping after epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"{target}: no finite checkpoint")
    model.load_state_dict(best_state)
    evaluations = {
        split_name: _evaluate(model, loaders[split_name], criterion, device)
        for split_name in ("train", "val", "test")
    }
    threshold = _optimal_threshold(
        evaluations["val"]["labels"], evaluations["val"]["probabilities"]
    )

    metric_rows = []
    prediction_rows = []
    for split_name, evaluation in evaluations.items():
        metrics = _binary_metrics(
            evaluation["labels"], evaluation["probabilities"], threshold
        )
        metric_rows.append({
            "target": target,
            "split": split_name,
            "status": "ok",
            **{f"metric_{key}": value for key, value in metrics.items()},
        })
        for local_index, row_index in enumerate(evaluation["row_indices"]):
            source = manifest.iloc[int(row_index)]
            probability = float(evaluation["probabilities"][local_index])
            prediction_rows.append({
                "target": target,
                "split": split_name,
                "sample_id": source["sample_id"],
                "event_type": source["event_type"],
                "base_event_id": source["base_event_id"],
                "frame_index": int(source["frame_index"]),
                "video_id": source["video_id"],
                "hospital_id": str(source["hospital_id"]),
                "y_true": int(evaluation["labels"][local_index]),
                "score": probability,
                "threshold": float(threshold),
                "y_pred": int(probability >= threshold),
                "match_delta_h": float(source["match_delta_h"]),
            })

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{target}_rgb20_nonadj.pt")
    torch.save(
        {
            "model_state_dict": best_state,
            "target": target,
            "threshold": threshold,
            "rgb_mean": mean,
            "rgb_std": std,
            "best_val_auc": best_score,
            "input_frames": "20 non-adjacent frames from the matched nearest video",
        },
        checkpoint_path,
    )
    return model, metric_rows, prediction_rows, history, [
        {"target": target, "hospital_id": hospital_id, "split": split_name}
        for split_name, hospital_ids in split.items()
        for hospital_id in sorted(hospital_ids)
    ]


def _aggregate_event_predictions(predictions_df):
    if predictions_df.empty:
        return predictions_df.copy()
    group_columns = ["target", "split", "base_event_id"]
    event_predictions = predictions_df.groupby(group_columns, as_index=False).agg(
        event_type=("event_type", "first"),
        video_id=("video_id", "first"),
        hospital_id=("hospital_id", "first"),
        y_true=("y_true", "first"),
        score=("score", "mean"),
        threshold=("threshold", "first"),
        match_delta_h=("match_delta_h", "first"),
        frame_count=("frame_index", "nunique"),
    )
    event_predictions["y_pred"] = (
        event_predictions["score"] >= event_predictions["threshold"]
    ).astype(int)
    return event_predictions


def _event_metric_rows(event_predictions):
    rows = []
    for (target, split), group in event_predictions.groupby(["target", "split"]):
        metrics = _binary_metrics(
            group["y_true"].to_numpy(), group["score"].to_numpy(),
            float(group["threshold"].iloc[0]),
        )
        rows.append({
            "target": target,
            "split": split,
            "status": "ok",
            **{f"metric_{key}": value for key, value in metrics.items()},
        })
    return pd.DataFrame(rows)


def train_and_evaluate(manifest, face, output_dir=OUTPUT_DIR):
    _set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(LOG_DIR, "loss_*.csv")):
        os.remove(path)
    for path in glob.glob(os.path.join(LOG_DIR, "training_history*.csv")):
        os.remove(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = _channel_statistics(face)
    parameter_count, _ = count_parameters(SingleFrameRGBNet())
    print(f"Device: {device}", flush=True)
    print(
        f"Model per task: SingleFrameRGBNet ({parameter_count:,} parameters; "
        "one non-adjacent RGB frame per augmented sample)",
        flush=True,
    )
    print(
        f"RGB normalization mean={mean.round(4).tolist()} std={std.round(4).tolist()}",
        flush=True,
    )

    all_metrics = []
    all_predictions = []
    all_history = []
    all_patient_splits = []
    for task_index, target in enumerate(TARGETS):
        print(f"\n{'=' * 68}\nTask [{task_index + 1}/{len(TARGETS)}]: {target}\n{'=' * 68}", flush=True)
        split = _patient_level_split_for_task(manifest, target, SEED + task_index)
        if split is None:
            row = {
                "target": target,
                "split": "test",
                "status": "skipped",
                "reason": "insufficient stratifiable patients",
            }
            all_metrics.append(row)
            print(f"  SKIP: {row['reason']}", flush=True)
            continue
        model, metrics, predictions, history, patient_splits = _train_one_task(
            face, manifest, target, split, mean, std, device
        )
        if isinstance(metrics, dict):
            all_metrics.append(metrics)
            print(f"  SKIP: {metrics['reason']}", flush=True)
            continue
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_history.extend(history)
        all_patient_splits.extend(patient_splits)
        test_metrics = [row for row in metrics if row["split"] == "test"]
        if test_metrics:
            test = test_metrics[0]
            print(
                f"  {target} test: "
                f"bACC={float(test.get('metric_balanced_accuracy', np.nan)):.4f} "
                f"AUC={float(test.get('metric_roc_auc', np.nan)):.4f}",
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame_metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    event_predictions_df = _aggregate_event_predictions(predictions_df)
    event_metrics_df = _event_metric_rows(event_predictions_df)
    skipped_df = frame_metrics_df[frame_metrics_df.get("status", "") == "skipped"]
    if len(skipped_df):
        event_metrics_df = pd.concat([event_metrics_df, skipped_df], ignore_index=True)
    history_df = pd.DataFrame(all_history)
    frame_metrics_df.to_csv(os.path.join(output_dir, "frame_metrics.csv"), index=False)
    event_metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    event_predictions_df.to_csv(
        os.path.join(output_dir, "event_predictions.csv"), index=False
    )
    pd.DataFrame(all_patient_splits).to_csv(
        os.path.join(output_dir, "patient_splits.csv"), index=False
    )
    if not history_df.empty:
        history_df.to_csv(os.path.join(LOG_DIR, "loss_all.csv"), index=False)
        history_df.to_csv(os.path.join(LOG_DIR, "training_history_all.csv"), index=False)
        for target in history_df["target"].unique():
            history_df[history_df["target"] == target].to_csv(
                os.path.join(LOG_DIR, f"loss_{target}.csv"), index=False
            )
        _plot_loss_curves(history_df, output_dir)

    test_rows = event_metrics_df[
        (event_metrics_df["split"] == "test")
        & (event_metrics_df["status"] == "ok")
        & event_metrics_df["metric_roc_auc"].notna()
    ]
    print("\nOVERALL TEST SUMMARY", flush=True)
    print(f"  Tasks evaluated: {len(test_rows)}", flush=True)
    if len(test_rows):
        print(
            f"  Macro bACC: {test_rows['metric_balanced_accuracy'].astype(float).mean():.4f}",
            flush=True,
        )
        print(
            f"  Macro AUC:  {test_rows['metric_roc_auc'].astype(float).mean():.4f}",
            flush=True,
        )
    return event_metrics_df, predictions_df


def main():
    parser = argparse.ArgumentParser(
        description="Train task-specific RGB models with 20 non-adjacent augmented frames"
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    arguments = parser.parse_args()
    features_path = os.path.join(arguments.output_dir, "features.npz")
    manifest_path = os.path.join(arguments.output_dir, "manifest.csv")
    if not os.path.exists(features_path) or not os.path.exists(manifest_path):
        print(f"ERROR: features.npz/manifest.csv not found under {arguments.output_dir}")
        sys.exit(1)
    data = np.load(features_path, allow_pickle=True)
    manifest = pd.read_csv(manifest_path, dtype={"hospital_id": str})
    train_and_evaluate(manifest, data["face"], output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
