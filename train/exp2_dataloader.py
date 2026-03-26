"""
Experiment 02 Dataloader
========================
Builds paired ECG/rPPG train-val loaders for signal-to-signal translation.

This module reuses the Experiment 01 data loading pipeline:
- mirror*_auto_cleaned discovery
- sliding windows
- resample to target length
- per-window Z-score normalization
- patient-level split by Hospital_Patient_ID
"""

import csv
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


# Add train/ to path so imports work when run from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_dataloader import BPDataset


class PairedSignalDataset(Dataset):
    """Wrap BPDataset and expose only paired ECG and rPPG windows."""

    def __init__(self, root_dir, window_sec=3.0, step_sec=1.0, target_length=1024):
        self.base = BPDataset(
            root_dir,
            window_sec=window_sec,
            step_sec=step_sec,
            target_length=target_length,
        )
        self.hospital_pids = self.base.hospital_pids
        self.lab_pids = self.base.lab_pids
        self.mirror_nums = self.base.mirror_nums

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ecg, rppg, _ = self.base[idx]
        return ecg, rppg


def build_paired_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=1024,
    debug=False,
):
    """
    Build train and validation loaders for paired ECG/rPPG translation.

    Returns:
        train_loader, val_loader
    """
    dataset = PairedSignalDataset(
        root_dir,
        window_sec=window_sec,
        step_sec=step_sec,
        target_length=target_length,
    )

    rng = np.random.default_rng(seed)
    unique_pids = list(dict.fromkeys(dataset.hospital_pids))
    rng.shuffle(unique_pids)

    if not unique_pids:
        raise RuntimeError("No valid patient windows found for Experiment 2.")

    n_val_pids = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val_pids])
    train_pids = set(unique_pids[n_val_pids:])

    train_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in train_pids]
    val_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in val_pids]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(
        "[Exp2 DataLoader] Patient-level split - "
        f"train: {len(train_pids)} patients / {len(train_indices)} samples, "
        f"val: {len(val_pids)} patients / {len(val_indices)} samples"
    )

    if debug:
        _save_split_report(dataset, train_indices, val_indices, root_dir)

    return train_loader, val_loader


def _save_split_report(
    dataset: PairedSignalDataset,
    train_indices: List[int],
    val_indices: List[int],
    save_dir: str,
):
    """Save Lab_Patient_ID split report with mirror number for debugging."""
    rows: List[Tuple[int, str, int, int]] = []

    def collect_rows(indices, split):
        key_counts = {}
        for i in indices:
            key = (dataset.lab_pids[i], dataset.mirror_nums[i])
            key_counts[key] = key_counts.get(key, 0) + 1
        for (pid, mirror), samples in sorted(key_counts.items()):
            rows.append((pid, split, samples, mirror))

    collect_rows(train_indices, "train")
    collect_rows(val_indices, "val")

    out_path = os.path.join(save_dir, "split_debug_exp2.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lab_Patient_ID", "Split", "Samples", "Mirror"])
        writer.writerows(rows)

    print(f"[Exp2 Split Debug] Saved to {out_path}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tr_loader, va_loader = build_paired_dataloaders(root, batch_size=8, debug=True)
    ecg_batch, rppg_batch = next(iter(tr_loader))
    print(f"ECG batch:  {tuple(ecg_batch.shape)}")
    print(f"rPPG batch: {tuple(rppg_batch.shape)}")
    print(f"Train batches: {len(tr_loader)}, Val batches: {len(va_loader)}")
