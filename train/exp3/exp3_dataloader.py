"""Experiment 03 dataloader for rPPG -> (Heart Rate, SpO2) multitask regression."""

import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset, Subset


HR_MIN = 30.0
HR_MAX = 200.0
SPO2_MIN = 70.0
SPO2_MAX = 100.0


def normalize_hr(x):
    return (x - HR_MIN) / (HR_MAX - HR_MIN)


def denormalize_hr(x):
    return x * (HR_MAX - HR_MIN) + HR_MIN


def normalize_spo2(x):
    return (x - SPO2_MIN) / (SPO2_MAX - SPO2_MIN)


def denormalize_spo2(x):
    return x * (SPO2_MAX - SPO2_MIN) + SPO2_MIN


def _zscore(x):
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


class VitalSignsDataset(Dataset):
    """Per-window rPPG samples with multitask labels and masks."""

    def __init__(
        self,
        root_dir,
        window_sec=3.0,
        step_sec=1.0,
        target_length=512,
        max_windows_per_patient=None,
        max_patients=None,
    ):
        self.samples = []
        self.hospital_pids = []

        mirror_dirs = sorted(glob.glob(os.path.join(root_dir, "mirror*_auto_cleaned")))
        if not mirror_dirs:
            raise FileNotFoundError(f"No mirror*_auto_cleaned directories found in {root_dir}")

        patient_count = 0
        stop_loading = False

        for mirror_dir in mirror_dirs:
            if stop_loading:
                break
            mirror_name = os.path.basename(mirror_dir)
            mirror_prefix = mirror_name.split("_")[0]  # mirror1
            mirror_num = int("".join(filter(str.isdigit, mirror_prefix)))
            merged_path = os.path.join(root_dir, f"merged_patient_info_{mirror_num}.csv")
            if not os.path.exists(merged_path):
                continue

            merged_df = pd.read_csv(merged_path)
            label_lookup = {}
            for _, row in merged_df.iterrows():
                lab_id = int(row["lab_patient_id"])
                label_lookup[lab_id] = {
                    "hospital_pid": row.get("hospital_patient_id", f"m{mirror_num}_lab{lab_id}"),
                    "heart_rate": float(row.get("heart_rate", -1)),
                    "blood_oxygen": float(row.get("blood_oxygen", -1)),
                }

            signal_files = sorted(glob.glob(os.path.join(mirror_dir, "patient_*.csv")))
            for fpath in signal_files:
                if max_patients is not None and patient_count >= max_patients:
                    stop_loading = True
                    break

                fname = os.path.basename(fpath)
                parts = fname.replace(".csv", "").split("_")
                try:
                    lab_pid = int(parts[1])
                except (IndexError, ValueError):
                    continue

                if lab_pid not in label_lookup:
                    continue

                labels = label_lookup[lab_pid]
                hr = labels["heart_rate"]
                spo2 = labels["blood_oxygen"]
                hr_valid = hr != -1
                spo2_valid = spo2 != -1
                if not hr_valid and not spo2_valid:
                    continue

                patient_count += 1

                try:
                    sig_df = pd.read_csv(fpath)
                except Exception:
                    continue

                if "Timestamp" not in sig_df.columns or "RPPG" not in sig_df.columns:
                    continue

                timestamps = sig_df["Timestamp"].values
                rppg = sig_df["RPPG"].values

                if len(timestamps) < 2:
                    continue

                dt = np.median(np.diff(timestamps))
                if dt <= 0:
                    continue
                fs = 1.0 / dt

                window_samples = int(window_sec * fs)
                step_samples = int(step_sec * fs)
                if window_samples > len(rppg):
                    continue

                patient_windows = 0
                start = 0
                while start + window_samples <= len(rppg):
                    rppg_win = rppg[start:start + window_samples].copy()
                    if np.isnan(rppg_win).any():
                        start += step_samples
                        continue

                    rppg_win = resample(rppg_win, target_length)
                    if np.isnan(rppg_win).any() or rppg_win.std() < 1e-6:
                        start += step_samples
                        continue

                    rppg_win = _zscore(rppg_win).astype(np.float32)

                    target = np.array([
                        normalize_hr(hr) if hr_valid else 0.0,
                        normalize_spo2(spo2) if spo2_valid else 0.0,
                    ], dtype=np.float32)
                    mask = np.array([
                        1.0 if hr_valid else 0.0,
                        1.0 if spo2_valid else 0.0,
                    ], dtype=np.float32)

                    self.samples.append((rppg_win, target, mask))
                    self.hospital_pids.append(labels["hospital_pid"])

                    patient_windows += 1
                    start += step_samples
                    if max_windows_per_patient is not None and patient_windows >= max_windows_per_patient:
                        break

        print(f"[Exp3 Dataset] Loaded {len(self.samples)} windows with HR/SpO2 labels.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rppg, target, mask = self.samples[idx]
        return (
            torch.from_numpy(rppg).unsqueeze(0),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )


def build_vitals_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=512,
    max_windows_per_patient=None,
    max_patients=None,
):
    dataset = VitalSignsDataset(
        root_dir,
        window_sec=window_sec,
        step_sec=step_sec,
        target_length=target_length,
        max_windows_per_patient=max_windows_per_patient,
        max_patients=max_patients,
    )

    rng = np.random.default_rng(seed)
    unique_pids = list(dict.fromkeys(dataset.hospital_pids))
    rng.shuffle(unique_pids)
    if not unique_pids:
        raise RuntimeError("No valid samples for Experiment 3.")

    n_val_pids = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val_pids])
    train_pids = set(unique_pids[n_val_pids:])

    train_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in train_pids]
    val_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in val_pids]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(
        "[Exp3 DataLoader] Patient-level split - "
        f"train: {len(train_pids)} patients / {len(train_indices)} samples, "
        f"val: {len(val_pids)} patients / {len(val_indices)} samples"
    )

    return train_loader, val_loader
