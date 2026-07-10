"""Train/evaluate per-task face-only binary classifiers for Exp2."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    AUGMENT_HORIZONTAL_FLIP,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LOG_DIR,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_EPOCHS,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    OUTPUT_DIR,
    POS_WEIGHT_MAX,
    SEED,
    TARGETS,
    WEIGHT_DECAY,
)
from .models import FaceOnlyCNN, count_parameters


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FaceDataset(Dataset):
    def __init__(self, face, labels, sample_id=None, hospital_id=None, augment=False):
        self.face = face
        self.labels = labels
        self.sample_id = sample_id
        self.hospital_id = hospital_id
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        face = self.face[idx]
        if self.augment and AUGMENT_HORIZONTAL_FLIP and np.random.rand() < 0.5:
            face = np.flip(face, axis=1).copy()
        return torch.from_numpy(face).unsqueeze(0), torch.tensor(float(self.labels[idx]))


def _patient_level_split_for_task(manifest, target_name, seed):
    col = pd.to_numeric(manifest[target_name], errors="coerce")
    valid = manifest.loc[col.notna()].copy()

    patient_rows = []
    for hospital_id, group in valid.groupby("hospital_id"):
        vals = pd.to_numeric(group[target_name], errors="coerce").dropna()
        if vals.empty:
            continue
        patient_rows.append({"hospital_id": str(hospital_id), "y": int(vals.max())})
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


def _filter_indices_by_split(manifest, face_np, target_name, split):
    col = pd.to_numeric(manifest[target_name], errors="coerce")
    valid_all = col.notna().to_numpy()
    hospital_ids = manifest["hospital_id"].astype(str).to_numpy()

    result = {}
    for sname in ["train", "val", "test"]:
        mask = np.array([hid in split[sname] for hid in hospital_ids]) & valid_all
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            result[sname] = None
        else:
            result[sname] = {
                "face": face_np[indices],
                "labels": col.iloc[indices].to_numpy(dtype=np.float32),
                "sample_id": manifest["sample_id"].iloc[indices].astype(str).to_numpy(),
                "hospital_id": manifest["hospital_id"].iloc[indices].astype(str).to_numpy(),
            }
    return result


def _binary_metrics(y_true, y_score, threshold=0.5):
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "accuracy": np.nan, "balanced_accuracy": np.nan,
            "f1": np.nan, "roc_auc": np.nan, "average_precision": np.nan,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0, "n": int(len(y_true)),
            "positive_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
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


def _make_loader(face, labels, batch_size, shuffle, augment=False):
    ds = FaceDataset(face, labels, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for face, labels in loader:
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(face)
        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_logits, all_labels = [], []
    for face, labels in loader:
        face = face.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        logits = model(face)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0))
    return total_loss / max(n_batches, 1), probs.numpy().ravel(), labels.numpy().ravel()


def _train_one_task(data_dict, target_name, device):
    train_y = data_dict["train"]["labels"]
    n_pos = int((train_y > 0.5).sum())
    n_neg = int((train_y < 0.5).sum())
    if n_pos < MIN_TRAIN_SAMPLES_PER_CLASS or n_neg < MIN_TRAIN_SAMPLES_PER_CLASS:
        return None, {"target": target_name, "split": "test", "status": "skipped",
                      "reason": f"insufficient class: pos={n_pos}, neg={n_neg}"}, [], []

    loaders = {}
    for sname in ["train", "val", "test"]:
        if data_dict[sname] is None:
            loaders[sname] = None
        else:
            d = data_dict[sname]
            loaders[sname] = _make_loader(
                d["face"], d["labels"], BATCH_SIZE, shuffle=(sname == "train"), augment=(sname == "train"))
    if loaders["val"] is None or loaders["test"] is None:
        return None, {"target": target_name, "split": "test", "status": "skipped",
                      "reason": "empty val/test split"}, [], []

    model = FaceOnlyCNN().to(device)
    pos_weight_value = min(POS_WEIGHT_MAX, max(1.0, n_neg / max(n_pos, 1)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_SCHEDULER_FACTOR,
                                  patience=LR_SCHEDULER_PATIENCE, min_lr=1e-6)

    best_val_bacc, patience_counter, best_state = -1.0, 0, None
    loss_log = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_probs, val_labels = _evaluate(model, loaders["val"], criterion, device)
        val_m = _binary_metrics(val_labels, val_probs)
        val_bacc = val_m["balanced_accuracy"]
        val_auc = val_m["roc_auc"]
        if np.isnan(val_bacc):
            val_bacc = -val_loss
        loss_log.append({
            "epoch": epoch,
            "target": target_name,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_bacc": float(val_bacc) if not np.isnan(val_bacc) else float("nan"),
            "val_roc_auc": float(val_auc) if not np.isnan(val_auc) else float("nan"),
        })
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

    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(os.path.join(LOG_DIR, f"loss_{target_name}.csv"), index=False)

    metrics, predictions = [], []
    for sname in ["train", "val", "test"]:
        if loaders[sname] is None:
            continue
        _, probs, labels = _evaluate(model, loaders[sname], criterion, device)
        m = _binary_metrics(labels, probs)
        metrics.append({"split": sname, "target": target_name,
                        **{f"metric_{k}": v for k, v in m.items()}})
        if sname == "test":
            d = data_dict[sname]
            for i in range(len(probs)):
                predictions.append({
                    "target": target_name,
                    "split": sname,
                    "sample_id": str(d["sample_id"][i]),
                    "hospital_id": str(d["hospital_id"][i]),
                    "y_true": int(labels[i]),
                    "score": float(probs[i]),
                })
    return model, metrics, predictions, loss_log


def _plot_loss_curves(loss_df, output_dir):
    tasks = loss_df["target"].unique()
    if len(tasks) == 0:
        return
    n_cols = min(3, len(tasks))
    n_rows = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_df = loss_df[loss_df["target"] == task]
        epochs = task_df["epoch"].to_numpy()
        ax.plot(epochs, task_df["train_loss"].to_numpy(), "b-", label="Train Loss", linewidth=1.5, alpha=0.7)
        ax.plot(epochs, task_df["val_loss"].to_numpy(), "r-", label="Val Loss", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(epochs, task_df["val_bacc"].to_numpy(), "g-", label="Val bACC", linewidth=1.5)
        ax2.plot(epochs, task_df["val_roc_auc"].to_numpy(), "m-", label="Val ROC-AUC", linewidth=1.5)
        ax2.set_ylabel("Score", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.set_ylim(-0.05, 1.05)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")
        ax.set_title(task, fontsize=9)

    for j in range(len(tasks), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Exp2 Face-Only: Per-Task Loss & Validation Metrics", fontsize=14, y=1.01)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss & metric curves saved to {plot_path}")


def train_and_evaluate(manifest, face, output_dir=OUTPUT_DIR):
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    total_p, _ = count_parameters(FaceOnlyCNN())
    print(f"Model: FaceOnlyCNN ({total_p:,} params per task)")
    all_metrics, all_predictions, all_loss_logs = [], [], []

    for t_idx, target_name in enumerate(TARGETS):
        print(f"\n{'='*60}\nTask [{t_idx+1}/{len(TARGETS)}]: {target_name}\n{'='*60}")
        split = _patient_level_split_for_task(manifest, target_name, SEED + t_idx)
        if split is None:
            print("  SKIP: insufficient patients")
            all_metrics.append({"target": target_name, "split": "test", "status": "skipped",
                                "reason": "insufficient patients"})
            continue
        data_dict = _filter_indices_by_split(manifest, face, target_name, split)
        if data_dict["train"] is None:
            print("  SKIP: no training samples")
            all_metrics.append({"target": target_name, "split": "test", "status": "skipped",
                                "reason": "no training samples"})
            continue
        n_train = len(data_dict["train"]["labels"])
        n_val = len(data_dict["val"]["labels"]) if data_dict["val"] else 0
        n_test = len(data_dict["test"]["labels"]) if data_dict["test"] else 0
        pos_rate = float(np.mean(data_dict["train"]["labels"]))
        print(f"  Samples: train={n_train} val={n_val} test={n_test} pos_rate={pos_rate:.2%}")
        model, metrics, predictions, loss_log = _train_one_task(data_dict, target_name, device)
        if isinstance(metrics, dict) and metrics.get("status") == "skipped":
            all_metrics.append(metrics)
            print(f"  SKIP: {metrics['reason']}")
            continue
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
        all_loss_logs.extend(loss_log)
        test_m = [m for m in metrics if m["split"] == "test"]
        if test_m:
            tm = test_m[0]
            print(f"  Test: bACC={float(tm.get('metric_balanced_accuracy', np.nan)):.3f} "
                  f"ROC-AUC={float(tm.get('metric_roc_auc', np.nan)):.3f}")
        if model is not None:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_{target_name}.pt"))

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    if all_loss_logs:
        loss_df = pd.DataFrame(all_loss_logs)
        loss_df.to_csv(os.path.join(LOG_DIR, "loss_all.csv"), index=False)
        _plot_loss_curves(loss_df, output_dir)

    if "metric_balanced_accuracy" in metrics_df.columns:
        test_rows = metrics_df[(metrics_df["split"] == "test") & metrics_df["metric_balanced_accuracy"].notna()]
        if len(test_rows):
            print("\nOVERALL SUMMARY")
            print(f"  Tasks evaluated: {len(test_rows)}")
            print(f"  Macro bACC: {test_rows['metric_balanced_accuracy'].astype(float).mean():.4f}")
            print(f"  Macro AUC:  {test_rows['metric_roc_auc'].astype(float).mean():.4f}")
    skipped = metrics_df[metrics_df.get("status", "") == "skipped"]
    if len(skipped):
        print("\nSkipped Tasks:")
        for _, r in skipped.iterrows():
            print(f"  - {r['target']}: {r.get('reason', 'unknown')}")
    return metrics_df, predictions_df


def main():
    parser = argparse.ArgumentParser(description="Exp2 face-only training/evaluation")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    features_path = os.path.join(args.output_dir, "features.npz")
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    if not os.path.exists(features_path) or not os.path.exists(manifest_path):
        print(f"ERROR: features.npz/manifest.csv not found under {args.output_dir}")
        sys.exit(1)
    data = np.load(features_path, allow_pickle=True)
    manifest = pd.read_csv(manifest_path, dtype=str)
    _ = train_and_evaluate(manifest, data["face"], output_dir=args.output_dir)


if __name__ == "__main__":
    main()
