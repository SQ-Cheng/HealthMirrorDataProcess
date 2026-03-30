"""Experiment 04-X dataloader: full-data SNR-ranked SQI regression dataset."""

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


def _mirror_glob_by_source(data_source):
    if data_source == "sqi":
        return "mirror*_auto_cleaned_sqi"
    if data_source == "cleaned":
        return "mirror*_auto_cleaned"
    raise ValueError(f"Unsupported data_source: {data_source}")


def compute_snr_db(x, fs):
    """Estimate physiological narrow-band SNR around dominant cardiac frequency."""
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


class RPPGSQIRankDataset(Dataset):
    """All-window dataset with SQI target defined by global SNR rank."""

    def __init__(
        self,
        root_dir,
        window_sec=3.0,
        step_sec=1.0,
        target_length=256,
        data_source="sqi",
        max_windows_per_patient=None,
        max_patients=None,
    ):
        self.windows = []
        self.snr_db = []
        self.sqi = []
        self.hospital_pids = []

        pattern = _mirror_glob_by_source(data_source)
        mirror_dirs = sorted(glob.glob(os.path.join(root_dir, pattern)))
        if not mirror_dirs:
            raise FileNotFoundError(f"No {pattern} directories found in {root_dir}")

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
                    start += step_samples

                    if np.isnan(win).any():
                        continue

                    win = resample(win, target_length)
                    if np.isnan(win).any() or win.std() < 1e-6:
                        continue
                    if (np.abs(win) < 1e-8).mean() > 0.95:
                        continue

                    win = _zscore(win)
                    fs_target = target_length / window_sec
                    snr = compute_snr_db(win, fs=fs_target)

                    self.windows.append(win.astype(np.float32))
                    self.snr_db.append(float(snr))
                    self.hospital_pids.append(hospital_pid)

                    patient_windows += 1
                    if max_windows_per_patient is not None and patient_windows >= max_windows_per_patient:
                        break

        if not self.windows:
            raise RuntimeError("No valid Exp4-X windows were loaded.")

        snr_np = np.array(self.snr_db)
        order = np.argsort(snr_np)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(order), dtype=np.float32)
        if len(order) > 1:
            sqi_np = ranks / float(len(order) - 1)
        else:
            sqi_np = np.zeros_like(ranks)
        self.sqi = sqi_np.tolist()

        print(
            f"[Exp4-X Dataset] Loaded {len(self.windows)} windows from {patient_count} patients. "
            f"SNR range: {snr_np.min():.2f} to {snr_np.max():.2f} dB"
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]).unsqueeze(0),
            torch.tensor(self.sqi[idx], dtype=torch.float32),
            torch.tensor(self.snr_db[idx], dtype=torch.float32),
        )


def _patient_split_indices(hospital_pids, val_ratio, seed):
    rng = np.random.default_rng(seed)
    unique_pids = list(dict.fromkeys(hospital_pids))
    rng.shuffle(unique_pids)
    if not unique_pids:
        raise RuntimeError("No valid patient IDs for Exp4-X split.")

    n_val = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val])
    train_pids = set(unique_pids[n_val:])

    train_indices = [i for i, pid in enumerate(hospital_pids) if pid in train_pids]
    val_indices = [i for i, pid in enumerate(hospital_pids) if pid in val_pids]
    if not train_indices or not val_indices:
        raise RuntimeError("Exp4-X split failed: empty train or validation set.")

    return train_indices, val_indices


def build_exp4x_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=256,
    data_source="sqi",
    max_windows_per_patient=None,
    max_patients=None,
    return_meta=False,
):
    dataset = RPPGSQIRankDataset(
        root_dir,
        window_sec=window_sec,
        step_sec=step_sec,
        target_length=target_length,
        data_source=data_source,
        max_windows_per_patient=max_windows_per_patient,
        max_patients=max_patients,
    )

    train_indices, val_indices = _patient_split_indices(dataset.hospital_pids, val_ratio, seed)

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

    train_snr = np.array([dataset.snr_db[i] for i in train_indices])
    val_snr = np.array([dataset.snr_db[i] for i in val_indices])
    print(
        "[Exp4-X DataLoader] Patient-level split - "
        f"train: {len(train_indices)}, val: {len(val_indices)} | "
        f"train SNR mean={train_snr.mean():.2f} dB, val SNR mean={val_snr.mean():.2f} dB"
    )

    if not return_meta:
        return train_loader, val_loader

    meta = {
        "dataset": dataset,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    return train_loader, val_loader, meta
