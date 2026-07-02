"""Training and evaluation pipeline for Exp2: multi-modal deep learning.

Key design:
    - Patient-level grouped split (hospital_id) to prevent data leakage.
    - Multi-task binary classification with BCEWithLogitsLoss.
    - Masked loss: tasks with missing labels are ignored per sample.
    - Early stopping on validation macro balanced accuracy.
    - Per-task and aggregate metrics reported.
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

# Minimum samples per CLASS in training set to include a task
_MIN_SAMPLES_PER_CLASS = 8
from .models import M3TNet, count_parameters


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
# PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════

class MultiModalDataset(Dataset):
    """PyTorch Dataset for ECG + Face multimodal data with multi-task labels.

    Each sample returns:
        ecg:   (1, ECG_LENGTH) float32 tensor
        face:  (1, FACE_SIZE, FACE_SIZE) float32 tensor
        labels: (num_active_tasks,) float32 tensor (NaN → 0)
        mask:  (num_active_tasks,) float32 tensor (1.0 = valid, 0.0 = missing)
    """

    def __init__(self, manifest, ecg, face, active_task_indices=None):
        self.manifest = manifest.reset_index(drop=True)
        self.ecg = ecg
        self.face = face
        all_targets = TARGETS
        if active_task_indices is not None:
            self.task_names = [all_targets[i] for i in active_task_indices]
        else:
            self.task_names = list(all_targets)

        # Build label matrix and mask for active tasks only
        labels_list = []
        mask_list = []
        for t in self.task_names:
            col = pd.to_numeric(self.manifest[t], errors="coerce")
            labels_list.append(col.to_numpy(dtype=np.float32))
            mask_list.append(col.notna().to_numpy(dtype=np.float32))
        self.labels = np.stack(labels_list, axis=1)   # (N, num_active)
        self.mask = np.stack(mask_list, axis=1)         # (N, num_active)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        ecg = torch.from_numpy(self.ecg[idx]).unsqueeze(0)           # (1, L)
        face = torch.from_numpy(self.face[idx]).unsqueeze(0)          # (1, H, W)
        labels = torch.from_numpy(self.labels[idx])                   # (T,)
        mask = torch.from_numpy(self.mask[idx])                       # (T,)
        # Replace NaN labels with 0 (masked out in loss anyway)
        labels = torch.nan_to_num(labels, nan=0.0)
        return ecg, face, labels, mask


# ═══════════════════════════════════════════════════════════════════════
# Data splitting
# ═══════════════════════════════════════════════════════════════════════

def _patient_level_split(manifest, seed):
    """Create train/val/test split grouped by hospital_id.

    Uses a composite label (OR of all target positives) for stratification.
    Returns dict: {'train': set(hospital_ids), 'val': set(...), 'test': set(...)}
    """
    patient_rows = []
    for hospital_id, group in manifest.groupby("hospital_id"):
        has_any_positive = 0
        for t in TARGETS:
            vals = pd.to_numeric(group[t], errors="coerce").dropna()
            if not vals.empty and int(vals.max()) == 1:
                has_any_positive = 1
                break
        patient_rows.append({"hospital_id": hospital_id, "y": has_any_positive})
    patients = pd.DataFrame(patient_rows)

    if patients["y"].nunique() < 2 or patients["y"].value_counts().min() < 3:
        # Fallback: random split
        rng = np.random.default_rng(seed)
        ids = patients["hospital_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(n * 0.20))
        n_val = max(1, int((n - n_test) * 0.20))
        return {
            "train": set(ids[:n - n_test - n_val]),
            "val": set(ids[n - n_test - n_val:n - n_test]),
            "test": set(ids[n - n_test:]),
        }

    # 60/20/20 split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train_idx, temp_idx = next(sss1.split(patients["hospital_id"], patients["y"]))
    temp = patients.iloc[temp_idx].reset_index(drop=True)

    if temp["y"].nunique() < 2 or temp["y"].value_counts().min() < 2:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    else:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_rel, test_rel = next(sss2.split(temp["hospital_id"], temp["y"]))

    return {
        "train": set(patients.iloc[train_idx]["hospital_id"].astype(str)),
        "val": set(temp.iloc[val_rel]["hospital_id"].astype(str)),
        "test": set(temp.iloc[test_rel]["hospital_id"].astype(str)),
    }


def _build_dataloaders(manifest, ecg, face, split, batch_size, active_task_indices=None):
    """Build train/val/test DataLoaders from a patient-level split."""
    hospital_ids = manifest["hospital_id"].astype(str)
    train_mask = hospital_ids.isin(split["train"]).to_numpy()
    val_mask = hospital_ids.isin(split["val"]).to_numpy()
    test_mask = hospital_ids.isin(split["test"]).to_numpy()

    datasets = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        indices = np.flatnonzero(mask)
        subset_manifest = manifest.iloc[indices].reset_index(drop=True)
        datasets[name] = MultiModalDataset(
            subset_manifest, ecg[indices], face[indices],
            active_task_indices=active_task_indices,
        )

    loaders = {}
    for name, ds in datasets.items():
        shuffle = (name == "train")
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=2, pin_memory=True,
        )
    return loaders, datasets


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def _binary_metrics(y_true, y_score, threshold=0.5):
    """Compute comprehensive binary classification metrics.

    Filters out NaN values from both predictions and labels.
    """
    # Remove NaN entries
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
# Training
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for ecg, face, labels, mask in loader:
        ecg = ecg.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(ecg, face)                          # (B, T)
        loss_per_task = criterion(logits, labels)          # (B, T)
        denom = mask.sum().clamp(min=1)
        masked_loss = (loss_per_task * mask).sum() / denom

        # Skip if loss is NaN/Inf (should not happen with proper data)
        if torch.isnan(masked_loss) or torch.isinf(masked_loss):
            continue

        masked_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += masked_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model: return loss + per-task predictions and labels."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits = []
    all_labels = []
    all_masks = []

    for ecg, face, labels, mask in loader:
        ecg = ecg.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(ecg, face)
        loss_per_task = criterion(logits, labels)          # (B, T)
        denom = mask.sum().clamp(min=1)
        masked_loss = (loss_per_task * mask).sum() / denom
        total_loss += masked_loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(mask.cpu())

    loss = total_loss / max(n_batches, 1)
    logits = torch.cat(all_logits, dim=0)   # (N, T)
    labels = torch.cat(all_labels, dim=0)   # (N, T)
    masks = torch.cat(all_masks, dim=0)     # (N, T)
    # NaN-safe sigmoid
    probs = torch.sigmoid(torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0))
    return loss, probs.numpy(), labels.numpy(), masks.numpy()


def _compute_val_score(val_metrics_per_task, val_loss=None):
    """Compute aggregate validation score: macro balanced accuracy over valid tasks.

    Falls back to negative val_loss if no task has valid metrics.
    """
    baccs = []
    for m in val_metrics_per_task:
        if m is not None and not np.isnan(m.get("balanced_accuracy", np.nan)):
            baccs.append(m["balanced_accuracy"])
    if baccs:
        return np.mean(baccs)
    # Fallback: use negative loss (lower loss = better)
    if val_loss is not None and not np.isnan(val_loss):
        return -val_loss
    return 0.0


def _filter_active_tasks(dataset, min_per_class=_MIN_SAMPLES_PER_CLASS):
    """Identify tasks with sufficient samples in BOTH classes.

    Returns:
        active_indices: list of task indices to keep.
        skipped: dict mapping task_name → reason string.
    """
    labels = dataset.labels   # (N, num_tasks)
    masks = dataset.mask      # (N, num_tasks)
    active = []
    skipped = {}
    for t_idx, t_name in enumerate(TARGETS):
        t_labels = labels[masks[:, t_idx] > 0.5, t_idx]
        n_pos = int((t_labels > 0.5).sum())
        n_neg = int((t_labels < 0.5).sum())
        if n_pos < min_per_class:
            skipped[t_name] = f"only {n_pos} positive (need ≥{min_per_class})"
        elif n_neg < min_per_class:
            skipped[t_name] = f"only {n_neg} negative (need ≥{min_per_class})"
        else:
            active.append(t_idx)
    return active, skipped


# ═══════════════════════════════════════════════════════════════════════
# Training orchestration
# ═══════════════════════════════════════════════════════════════════════


def train_and_evaluate(manifest, ecg, face, output_dir=OUTPUT_DIR):
    """Full training + evaluation pipeline for the multi-modal model.

    Args:
        manifest: DataFrame with sample metadata and labels.
        ecg: (N, ECG_LENGTH) float32 array.
        face: (N, FACE_SIZE, FACE_SIZE) float32 array.
        output_dir: Output directory for results.

    Returns:
        metrics_df: Per-task metrics DataFrame.
        predictions_df: Per-sample prediction scores.
    """
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Split data ──────────────────────────────────────────────────
    print("\nSplitting data (patient-level) ...")
    split = _patient_level_split(manifest, SEED)
    for name in ["train", "val", "test"]:
        print(f"  {name}: {len(split[name])} patients")

    # ── First pass: build train dataset with ALL tasks to filter ─────
    hospital_ids = manifest["hospital_id"].astype(str)
    train_mask_all = hospital_ids.isin(split["train"]).to_numpy()
    train_indices = np.flatnonzero(train_mask_all)
    train_manifest_all = manifest.iloc[train_indices].reset_index(drop=True)
    train_ds_all = MultiModalDataset(train_manifest_all, ecg[train_indices],
                                     face[train_indices], active_task_indices=None)

    active_tasks, skipped_tasks = _filter_active_tasks(train_ds_all)
    active_task_names = [TARGETS[i] for i in active_tasks]
    num_active = len(active_tasks)
    print(f"\nActive tasks: {num_active}/{len(TARGETS)}")
    if skipped_tasks:
        print("  Skipped tasks:")
        for name, reason in skipped_tasks.items():
            print(f"    - {name}: {reason}")

    if num_active == 0:
        print("ERROR: No active tasks remaining after filtering. Aborting.")
        return None, None

    # ── Rebuild dataloaders with only active tasks ───────────────────
    loaders, datasets = _build_dataloaders(manifest, ecg, face, split, BATCH_SIZE,
                                           active_task_indices=active_tasks)
    print(f"  Train samples: {len(datasets['train'])}")
    print(f"  Val samples:   {len(datasets['val'])}")
    print(f"  Test samples:  {len(datasets['test'])}")

    # ── Build model ─────────────────────────────────────────────────
    model = M3TNet(num_tasks=num_active).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f"\nModel: {total_p:,} total params ({trainable_p:,} trainable)")

    # ── Loss, optimizer, scheduler ──────────────────────────────────
    pos_weight = torch.full((num_active,), POS_WEIGHT, device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  factor=LR_SCHEDULER_FACTOR,
                                  patience=LR_SCHEDULER_PATIENCE,
                                  min_lr=1e-6)

    # ── Training loop ───────────────────────────────────────────────
    print(f"\nTraining (max {MAX_EPOCHS} epochs, patience={EARLY_STOPPING_PATIENCE}) ...")
    best_val_score = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_score": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion,
                                 device, GRAD_CLIP_NORM)
        val_loss, val_probs, val_labels, val_masks = evaluate(
            model, loaders["val"], criterion, device
        )

        # Per-task val metrics (only for active tasks)
        val_metrics_per_task = []
        for t in range(num_active):
            t_mask = val_masks[:, t] > 0.5
            if t_mask.sum() < 1:
                val_metrics_per_task.append(None)
                continue
            y_t = val_labels[t_mask, t].astype(int)
            p_t = val_probs[t_mask, t]
            if len(np.unique(y_t)) < 2:
                val_metrics_per_task.append(None)
                continue
            val_metrics_per_task.append(_binary_metrics(y_t, p_t))

        val_score = _compute_val_score(val_metrics_per_task, val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)

        # Scheduler step
        scheduler.step(val_score)

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_bacc={val_score:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_score": val_score,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBest epoch: {best_epoch} (val_bacc={best_val_score:.4f})")

    # ── Load best model ─────────────────────────────────────────────
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Evaluate on test set ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    _, test_probs, test_labels, test_masks = evaluate(
        model, loaders["test"], criterion, device
    )

    metrics_rows = []
    prediction_rows = []

    # Also get train/val predictions for record
    for split_name, loader in [("train", loaders["train"]),
                                ("val", loaders["val"]),
                                ("test", loaders["test"])]:
        _, probs, labels, masks = evaluate(model, loader, criterion, device)
        ds = datasets[split_name]
        for t_idx, t_name in enumerate(active_task_names):
            t_mask = masks[:, t_idx] > 0.5
            if t_mask.sum() < 1:
                continue
            y_t = labels[t_mask, t_idx].astype(int)
            p_t = probs[t_mask, t_idx]
            if len(np.unique(y_t)) < 2:
                continue
            m = _binary_metrics(y_t, p_t)
            metrics_rows.append({
                "split": split_name,
                "target": t_name,
                **{f"metric_{k}": v for k, v in m.items()},
            })

            # Per-sample predictions (test only)
            if split_name == "test":
                t_mask_indices = np.flatnonzero(t_mask)
                for local_i, global_i in enumerate(t_mask_indices):
                    prediction_rows.append({
                        "target": t_name,
                        "sample_id": ds.manifest.iloc[global_i]["sample_id"],
                        "hospital_id": ds.manifest.iloc[global_i]["hospital_id"],
                        "y_true": int(y_t[local_i]),
                        "score": float(p_t[local_i]),
                    })

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    # Save
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    with open(os.path.join(output_dir, "split.json"), "w") as f:
        json.dump({k: sorted(list(v)) for k, v in split.items()}, f, indent=2)
    with open(os.path.join(LOG_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Print summary ───────────────────────────────────────────────
    print("\nTest Set Summary (macro-averaged over valid tasks):")
    test_metrics = metrics_df[metrics_df["split"] == "test"]
    for metric_name in ["metric_balanced_accuracy", "metric_roc_auc",
                         "metric_f1", "metric_average_precision"]:
        vals = test_metrics[metric_name].dropna()
        if len(vals) > 0:
            print(f"  {metric_name.replace('metric_', '')}: {vals.mean():.4f} ± {vals.std():.4f}")

    if skipped_tasks:
        print("\nSkipped Tasks (insufficient per-class samples in training set):")
        for name, reason in skipped_tasks.items():
            print(f"  - {name}: {reason}")

    print("\nPer-Task Test Results:")
    for t in active_task_names:
        row = test_metrics[test_metrics["target"] == t]
        if len(row) == 0:
            print(f"  {t}: SKIPPED (insufficient test samples)")
        else:
            r = row.iloc[0]
            print(f"  {t}: bACC={r['metric_balanced_accuracy']:.3f} | "
                  f"ROC-AUC={r['metric_roc_auc']:.3f} | "
                  f"F1={r['metric_f1']:.3f} | "
                  f"n={int(r['metric_n'])} | pos_rate={r['metric_positive_rate']:.2%}")

    return metrics_df, predictions_df


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp2: Train and evaluate multi-modal DL model")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    # Load features
    features_path = os.path.join(args.output_dir, "features.npz")
    if not os.path.exists(features_path):
        print(f"ERROR: features.npz not found at {features_path}")
        print("Run build_dataset.py first.")
        sys.exit(1)

    data = np.load(features_path, allow_pickle=True)
    manifest = pd.read_csv(os.path.join(args.output_dir, "manifest.csv"), dtype=str)

    ecg = data["ecg"]     # (N, L)
    face = data["face"]   # (N, H, W)

    print(f"Loaded {len(manifest)} samples, {manifest['hospital_id'].nunique()} patients")
    print(f"ECG shape: {ecg.shape}, Face shape: {face.shape}")

    _ = train_and_evaluate(manifest, ecg, face, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
