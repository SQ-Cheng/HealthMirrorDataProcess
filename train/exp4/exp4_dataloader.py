"""Experiment 04 dataloader for unsupervised rPPG artifact detection with autoencoders."""

import glob
import os

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


def compute_snr_db(x, fs):
    """Estimate narrow-band physiological SNR around dominant cardiac frequency."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2

    band = (freqs >= 0.5) & (freqs <= 5.0)
    if not np.any(band):
        return -100.0

    band_spec = spec[band]
    band_freqs = freqs[band]
    peak_freq = band_freqs[np.argmax(band_spec)]

    signal_band = (freqs >= (peak_freq - 0.15)) & (freqs <= (peak_freq + 0.15))
    signal_power = spec[signal_band].sum()
    noise_power = spec[band].sum() - signal_power
    noise_power = max(noise_power, 1e-12)

    return 10.0 * np.log10(max(signal_power, 1e-12) / noise_power)


class RPPGWindowDataset(Dataset):
    def __init__(
        self,
        root_dir,
        window_sec=3.0,
        step_sec=1.0,
        target_length=512,
        max_windows_per_patient=None,
        max_patients=None,
    ):
        self.windows = []
        self.snr_db = []
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
            mirror_prefix = mirror_name.split("_")[0]
            mirror_num = int("".join(filter(str.isdigit, mirror_prefix)))

            hospital_lookup = {}
            info_path = os.path.join(mirror_dir, "cleaned_patient_info.csv")
            if os.path.exists(info_path):
                info_df = pd.read_csv(info_path)
                for _, row in info_df.iterrows():
                    hospital_lookup[int(row["Lab_Patient_ID"])] = row["Hospital_Patient_ID"]

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
                hospital_pid = hospital_lookup.get(lab_pid, f"m{mirror_num}_lab{lab_pid}")
                start = 0
                while start + window_samples <= len(rppg):
                    win = rppg[start:start + window_samples].copy()
                    if np.isnan(win).any():
                        start += step_samples
                        continue

                    win = resample(win, target_length)
                    if np.isnan(win).any() or win.std() < 1e-6:
                        start += step_samples
                        continue
                    if (np.abs(win) < 1e-8).mean() > 0.95:
                        start += step_samples
                        continue

                    win = _zscore(win)
                    fs_target = target_length / window_sec
                    snr = compute_snr_db(win, fs=fs_target)

                    self.windows.append(win.astype(np.float32))
                    self.snr_db.append(float(snr))
                    self.hospital_pids.append(hospital_pid)

                    patient_windows += 1
                    start += step_samples
                    if max_windows_per_patient is not None and patient_windows >= max_windows_per_patient:
                        break

        print(f"[Exp4 Dataset] Loaded {len(self.windows)} rPPG windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]).unsqueeze(0), torch.tensor(self.snr_db[idx], dtype=torch.float32)


def build_artifact_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=512,
    clean_percentile=90.0,
    max_windows_per_patient=None,
    max_patients=None,
):
    dataset = RPPGWindowDataset(
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
        raise RuntimeError("No valid windows for Experiment 4.")

    n_val_pids = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val_pids])
    train_pids = set(unique_pids[n_val_pids:])

    train_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in train_pids]
    val_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in val_pids]

    train_snr = np.array([dataset.snr_db[i] for i in train_indices])
    threshold = float(np.percentile(train_snr, clean_percentile))
    train_clean_indices = [i for i in train_indices if dataset.snr_db[i] >= threshold]

    train_loader = DataLoader(
        Subset(dataset, train_clean_indices),
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
        "[Exp4 DataLoader] Patient-level split - "
        f"train all: {len(train_indices)}, train clean: {len(train_clean_indices)}, "
        f"val all: {len(val_indices)}, clean threshold={threshold:.2f} dB"
    )

    return train_loader, val_loader, threshold
