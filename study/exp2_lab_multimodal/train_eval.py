"""Training and evaluation pipeline for Exp2: per-task binary deep learning models.

Key design:
    - ONE BinaryM3TNet model is trained per lab-test task (not multi-head).
    - Each model uses ALL samples that have a valid label for that task.
    - Patient-level grouped split (hospital_id) to prevent data leakage.
    - Early stopping on validation balanced accuracy.
"""

import argparse
import json
import os
import sys

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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    FACE_SIZE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LOG_DIR,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    OUTPUT_DIR,
    POS_WEIGHT,
    SEED,
    TARGETS,
    WEIGHT_DECAY,
)
from .models import BinaryM3TNet, count_parameters


# ═══════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════

def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Dataset (single-task, array-based)
# ═══════════════════════════════════════════════════════════════════════

class ArrayDataset(Dataset):
    """Lightweight Dataset from numpy arrays (ECG, face, labels)."""

    def __init__(self, ecg, face, labels):
        self.ecg = ecg
        self.face = face
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.ecg[idx]).unsqueeze(0),
            torch.from_numpy(self.face[idx]).unsqueeze(0),
            torch.tensor(float(self.labels[idx])),
        )


# ═══════════════════════════════════════════════════════════════════════
# Data splitting
# ═══════════════════════════════════════════════════════════════════════

def _patient_level_split_for_task(manifest, target_name, seed):
    """Create train/val/test split grouped by hospital_id for a single task."""
    col = pd.to_numeric(manifest[target_name], errors="coerce")
    valid = manifest.loc[col.notna()].copy()

    patient_rows = []
    for hospital_id, group in valid.groupby("hospital_id"):
        vals = pd.to_numeric(group[target_name], errors="coerce").dropna()
        if vals.empty:
            continue
        patient_rows.append({"hospital_id": hospital_id, "y": int(vals.max())})
    patients = pd.DataFrame(patient_rows)

    if len(patients) < 5:
        return None

    if patients["y"].nunique() < 2 or patients["y"].value_counts().min() < 3:
        rng = np.random.default_rng(seed)
        ids = patients["hospital_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(n * 0.20))
        n_val = max(1, int((n - n_test) * 0.20))
        if n - n_test - n_val < 2:
            return None
        return {
            "train": set(ids[:n - n_test - n_val]),
            "val": set(ids[n - n_test - n_val:n - n_test]),
            "test": set(ids[n - n_test:]),
        }

    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
        train_idx, temp_idx = next(sss1.split(patients["hospital_id"], patients["y"]))
        temp = patients.iloc[temp_idx].reset_index(drop=True)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
        val_rel, test_rel = next(sss2.split(temp["hospital_id"], temp["y"]))
    except ValueError:
        return None

    return {
        "train": set(patients.iloc[train_idx]["hospital_id"].astype(str)),
        "val": set(temp.iloc[val_rel]["hospital_id"].astype(str)),
        "test": set(temp.iloc[test_rel]["hospital_id"].astype(str)),
    }


def _filter_indices_by_split(manifest, ecg_np, face_np, target_name, split):
    """Return dict: {'train': {ecg, face, labels}, 'val': ..., 'test': ...}."""
    col = pd.to_numeric(manifest[target_name], errors="coerce")
    valid_all = col.notna().to_numpy()
    hospital_ids = manifest["hospital_id"].astype(str).to_numpy()

    result = {}
    for sname in ["train", "val", "test"]:
        mask = np.array([hid in split[sname] for hid in hospital_ids])
        mask = mask & valid_all
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            result[sname] = None
        else:
            result[sname] = {
                "ecg": ecg_np[indices],
                "face": face_np[indices],
                "labels": col.iloc[indices].to_numpy(dtype=np.float32),
            }
    return result


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def _binary_metrics(y_true, y_score, threshold=0.5):
    """Compute comprehensive binary classification metrics. NaN-safe."""
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "accuracy": np.nan, "balanced_accuracy": np.nan,
            "f1": np.nan, "roc_auc": np.nan, "average_precision": np.nan,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0, "n": int(len(y_true)),
            "positive_rate": float(np.mean(y_true)) if len(y_true) > 0 else np.nan,
        }
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════

def _make_loader(ecg, face, labels, batch_size, shuffle):
    ds = ArrayDataset(ecg, face, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def _train_epoch_binary(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for ecg, face, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(ecg, face)
        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate_binary(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits, all_labels = [], []
    for ecg, face, labels in loader:
        ecg = ecg.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        logits = model(ecg, face)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    loss = total_loss / max(n_batches, 1)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0))
    return loss, probs.numpy().ravel(), labels.numpy().ravel()


# ═══════════════════════════════════════════════════════════════════════
# Per-task training
# ═══════════════════════════════════════════════════════════════════════

_MIN_SAMPLES_PER_CLASS = 8


def _train_one_task(labels_dict, target_name, device):
    """Train a single BinaryM3TNet for one task.

    Returns: (model, metrics_list, predictions_list) or (None, skip_dict, [])
    """
    train_y = labels_dict["train"]["labels"]
    n_pos = int((train_y > 0.5).sum())
    n_neg = int((train_y < 0.5).sum())
    if n_pos < _MIN_SAMPLES_PER_CLASS or n_neg < _MIN_SAMPLES_PER_CLASS:
        return None, {
            "target": target_name, "split": "test", "status": "skipped",
            "reason": f"insufficient class: pos={n_pos}, neg={n_neg}",
        }, []

    # Build loaders
    loaders = {}
    for sname in ["train", "val", "test"]:
        if labels_dict[sname] is None:
            loaders[sname] = None
            continue
        d = labels_dict[sname]
        loaders[sname] = _make_loader(d["ecg"], d["face"], d["labels"],
                                      BATCH_SIZE, shuffle=(sname == "train"))

    # Model, loss, optimizer
    model = BinaryM3TNet().to(device)
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_SCHEDULER_FACTOR,
                                  patience=LR_SCHEDULER_PATIENCE, min_lr=1e-6)

    best_val_bacc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _train_epoch_binary(model, loaders["train"], optimizer,
                                         criterion, device, GRAD_CLIP_NORM)
        val_loss, val_probs, val_labels = _evaluate_binary(
            model, loaders["val"], criterion, device)
        val_m = _binary_metrics(val_labels, val_probs)
        val_bacc = val_m["balanced_accuracy"]
        if np.isnan(val_bacc):
            val_bacc = -val_loss

        scheduler.step(val_bacc)

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate all splits
    metrics = []
    predictions = []
    for sname in ["train", "val", "test"]:
        if loaders[sname] is None:
            continue
        _, probs, labels = _evaluate_binary(model, loaders[sname], criterion, device)
        m = _binary_metrics(labels, probs)
        metrics.append({"split": sname, "target": target_name,
                        **{f"metric_{k}": v for k, v in m.items()}})
        if sname == "test":
            for i in range(len(probs)):
                predictions.append({
                    "target": target_name, "y_true": int(labels[i]),
                    "score": float(probs[i]),
                })

    return model, metrics, predictions


# ═══════════════════════════════════════════════════════════════════════
# Main orchestration
# ═══════════════════════════════════════════════════════════════════════

def train_and_evaluate(manifest, ecg, face, output_dir=OUTPUT_DIR):
    """Train one BinaryM3TNet per task and collect results."""
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    total_p, _ = count_parameters(BinaryM3TNet())
    print(f"Model: BinaryM3TNet ({total_p:,} params per task)")
    print(f"Strategy: train ONE model per task ({len(TARGETS)} tasks)")

    all_metrics = []
    all_predictions = []

    for t_idx, target_name in enumerate(TARGETS):
        print(f"\n{'='*60}")
        print(f"Task [{t_idx+1}/{len(TARGETS)}]: {target_name}")
        print(f"{'='*60}")

        split = _patient_level_split_for_task(manifest, target_name, SEED + t_idx)
        if split is None:
            print(f"  SKIP: insufficient patients")
            all_metrics.append({"target": target_name, "split": "test",
                                "status": "skipped", "reason": "insufficient patients"})
            continue

        labels_dict = _filter_indices_by_split(manifest, ecg, face, target_name, split)
        if labels_dict["train"] is None:
            print(f"  SKIP: no training samples")
            continue

        n_train = len(labels_dict["train"]["labels"])
        n_val = len(labels_dict["val"]["labels"]) if labels_dict["val"] else 0
        n_test = len(labels_dict["test"]["labels"]) if labels_dict["test"] else 0
        pos_rate = float(np.mean(labels_dict["train"]["labels"]))
        print(f"  Samples: train={n_train} val={n_val} test={n_test}  pos_rate={pos_rate:.2%}")

        model, metrics, predictions = _train_one_task(labels_dict, target_name, device)

        if isinstance(metrics, dict) and metrics.get("status") == "skipped":
            all_metrics.append(metrics)
            print(f"  SKIP: {metrics['reason']}")
            continue

        for m in metrics:
            all_metrics.append(m)
        all_predictions.extend(predictions)

        test_m = [m for m in metrics if m["split"] == "test"]
        if test_m:
            tm = test_m[0]
            bacc = tm.get("metric_balanced_accuracy", float('nan'))
            auc = tm.get("metric_roc_auc", float('nan'))
            if isinstance(bacc, float):
                print(f"  Test: bACC={bacc:.3f}  ROC-AUC={auc:.3f}")

        if model is not None:
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, f"model_{target_name}.pt"))

    # ── Compile ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    test_rows = metrics_df[(metrics_df["split"] == "test") &
                           (metrics_df["metric_balanced_accuracy"].notna())]
    if len(test_rows) > 0:
        baccs = test_rows["metric_balanced_accuracy"].astype(float)
        aucs = test_rows["metric_roc_auc"].astype(float)
        f1s = test_rows["metric_f1"].astype(float)
        print(f"  Tasks evaluated: {len(test_rows)}")
        print(f"  Macro bACC:  {baccs.mean():.4f} ± {baccs.std():.4f}")
        print(f"  Macro AUC:   {aucs.mean():.4f} ± {aucs.std():.4f}")
        print(f"  Macro F1:    {f1s.mean():.4f} ± {f1s.std():.4f}")

    print("\nPer-Task Test Results:")
    for _, r in test_rows.iterrows():
        print(f"  {r['target']:30s} bACC={float(r['metric_balanced_accuracy']):.3f}  "
              f"ROC-AUC={float(r['metric_roc_auc']):.3f}  F1={float(r['metric_f1']):.3f}  "
              f"n={int(r['metric_n'])}  pos={float(r['metric_positive_rate']):.2%}")

    skipped = metrics_df[metrics_df.get("status", "") == "skipped"]
    if len(skipped) > 0:
        print("\nSkipped Tasks:")
        for _, r in skipped.iterrows():
            print(f"  - {r['target']}: {r.get('reason', 'unknown')}")

    return metrics_df, predictions_df


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp2: Per-task binary DL models")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    features_path = os.path.join(args.output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"ERROR: features.npz not found at {features_path}")
        sys.exit(1)

    data = np.load(features_path, allow_pickle=True)
    manifest = pd.read_csv(os.path.join(args.output_dir, "manifest.csv"), dtype=str)
    ecg = data["ecg"]
    face = data["face"]

    print(f"Loaded {len(manifest)} samples, {manifest['hospital_id'].nunique()} patients")
    _ = train_and_evaluate(manifest, ecg, face, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
