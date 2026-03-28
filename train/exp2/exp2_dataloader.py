"""Build paired ECG/rPPG train-val loaders for signal-to-signal translation."""

import csv
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset, Subset


def _zscore(x):
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


def _mirror_glob_by_source(data_source):
    if data_source == "sqi":
        return "mirror*_auto_cleaned_sqi"
    if data_source == "cleaned":
        return "mirror*_auto_cleaned"
    raise ValueError(f"Unsupported data_source: {data_source}")


class PairedSignalDataset(Dataset):
    """Paired ECG/rPPG windows without any BP-label dependency."""

    def __init__(
        self,
        root_dir,
        window_sec=3.0,
        step_sec=1.0,
        target_length=1024,
        data_source="sqi",
        max_windows_per_patient=None,
        max_patients=None,
    ):
        self.samples = []
        self.hospital_pids = []
        self.lab_pids = []
        self.mirror_nums = []

        pattern = _mirror_glob_by_source(data_source)
        mirror_dirs = sorted(glob.glob(os.path.join(root_dir, pattern)))
        if not mirror_dirs:
            raise FileNotFoundError(f"No {pattern} directories found in {root_dir}")

        patient_count = 0
        stop_loading = False

        for mirror_dir in mirror_dirs:
            if stop_loading:
                break
            dir_name = os.path.basename(mirror_dir)
            try:
                mirror_num = int("".join(filter(str.isdigit, dir_name.split("_")[0])))
            except (ValueError, IndexError):
                mirror_num = -1

            info_path = os.path.join(mirror_dir, "cleaned_patient_info.csv")
            hospital_lookup = {}
            if os.path.exists(info_path):
                info_df = pd.read_csv(info_path)
                for _, row in info_df.iterrows():
                    lab_id = int(row["Lab_Patient_ID"])
                    hospital_lookup[lab_id] = row["Hospital_Patient_ID"]

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

                try:
                    sig_df = pd.read_csv(fpath)
                except Exception:
                    continue

                patient_count += 1

                if "Timestamp" not in sig_df.columns or "RPPG" not in sig_df.columns or "ECG" not in sig_df.columns:
                    continue

                timestamps = sig_df["Timestamp"].values
                ecg = sig_df["ECG"].values
                rppg = sig_df["RPPG"].values

                if len(timestamps) < 2:
                    continue

                dt = np.median(np.diff(timestamps))
                if dt <= 0:
                    continue
                fs = 1.0 / dt

                window_samples = int(window_sec * fs)
                step_samples = int(step_sec * fs)
                if window_samples > len(ecg):
                    continue

                hospital_pid = hospital_lookup.get(lab_pid, f"m{mirror_num}_lab{lab_pid}")
                patient_windows = 0
                start = 0
                while start + window_samples <= len(ecg):
                    ecg_win = ecg[start:start + window_samples].copy()
                    rppg_win = rppg[start:start + window_samples].copy()

                    if np.isnan(ecg_win).any() or np.isnan(rppg_win).any():
                        start += step_samples
                        continue

                    ecg_win = resample(ecg_win, target_length)
                    rppg_win = resample(rppg_win, target_length)

                    if np.isnan(ecg_win).any() or np.isnan(rppg_win).any():
                        start += step_samples
                        continue

                    # Reject almost-flat windows and obvious zero artifacts.
                    if ecg_win.std() < 1e-6 or rppg_win.std() < 1e-6:
                        start += step_samples
                        continue
                    if (np.abs(rppg_win) < 1e-8).mean() > 0.95:
                        start += step_samples
                        continue

                    ecg_win = _zscore(ecg_win).astype(np.float32)
                    rppg_win = _zscore(rppg_win).astype(np.float32)

                    self.samples.append((ecg_win, rppg_win))
                    self.hospital_pids.append(hospital_pid)
                    self.lab_pids.append(lab_pid)
                    self.mirror_nums.append(mirror_num)

                    patient_windows += 1
                    start += step_samples

                    if max_windows_per_patient is not None and patient_windows >= max_windows_per_patient:
                        break

        print(f"[Exp2 Dataset] Loaded {len(self.samples)} paired windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg, rppg = self.samples[idx]
        return torch.from_numpy(ecg).unsqueeze(0), torch.from_numpy(rppg).unsqueeze(0)


def build_paired_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=1024,
    data_source="sqi",
    debug=False,
    max_windows_per_patient=None,
    max_patients=None,
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
        data_source=data_source,
        max_windows_per_patient=max_windows_per_patient,
        max_patients=max_patients,
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
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tr_loader, va_loader = build_paired_dataloaders(
        root,
        batch_size=8,
        debug=True,
        max_windows_per_patient=8,
    )
    ecg_batch, rppg_batch = next(iter(tr_loader))
    print(f"ECG batch:  {tuple(ecg_batch.shape)}")
    print(f"rPPG batch: {tuple(rppg_batch.shape)}")
    print(f"Train batches: {len(tr_loader)}, Val batches: {len(va_loader)}")
