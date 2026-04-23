"""Shared dataloader for single-signal masked reconstruction (Exp3 ECG/rPPG split)."""

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


def _rank01(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return (ranks / float(len(values) - 1)).astype(np.float32)


def _mirror_glob_by_source(data_source):
    if data_source == "sqi":
        return "mirror*_auto_cleaned_sqi"
    if data_source == "cleaned":
        return "mirror*_auto_cleaned"
    raise ValueError(f"Unsupported data_source: {data_source}")


def compute_snr_db(x, fs, lo_hz, hi_hz, peak_width_hz):
    """Estimate narrow-band SNR around dominant frequency in a target band."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2

    hi_hz = min(hi_hz, float(freqs[-1]) - 1e-6)
    if hi_hz <= lo_hz:
        return -100.0

    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band):
        return -100.0

    band_spec = spec[band]
    band_freqs = freqs[band]
    peak_freq = band_freqs[np.argmax(band_spec)]

    signal_band = (freqs >= (peak_freq - peak_width_hz)) & (freqs <= (peak_freq + peak_width_hz))
    signal_power = spec[signal_band].sum()
    noise_power = spec[band].sum() - signal_power
    noise_power = max(noise_power, 1e-12)

    return 10.0 * np.log10(max(signal_power, 1e-12) / noise_power)


def ecg_sqi_autocorr(ecg, fs):
    """Autocorrelation periodicity SQI in [0,1]."""
    x = ecg - np.mean(ecg)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(x) - 1:]
    if acf[0] <= 1e-12:
        return 0.0
    acf = acf / acf[0]

    lag_lo = max(1, int(fs * 0.33))
    lag_hi = min(len(acf) - 1, int(fs * 1.50))
    if lag_hi <= lag_lo:
        return 0.0

    peak = float(np.max(acf[lag_lo:lag_hi + 1]))
    return float(np.clip(peak, 0.0, 1.0))


class SingleSignalMaskedReconDataset(Dataset):
    """Per-window single-signal samples for masked reconstruction."""

    def __init__(
        self,
        root_dir,
        signal_type="ecg",
        window_sec=3.0,
        step_sec=1.0,
        target_length=256,
        data_source="sqi",
        max_windows_per_patient=None,
        max_patients=None,
    ):
        self.signal_type = signal_type.lower().strip()
        if self.signal_type not in {"ecg", "rppg"}:
            raise ValueError("signal_type must be 'ecg' or 'rppg'")

        self.samples = []
        self.hospital_pids = []
        self.source_records = []

        self.rppg_snr_db = []
        self.ecg_autocorr_sqi = []
        self.raw_quality = []
        self.quality_score = []

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

                required_cols = {"Timestamp", "RPPG", "ECG"}
                if not required_cols.issubset(set(sig_df.columns)):
                    continue

                timestamps = sig_df["Timestamp"].values
                rppg = sig_df["RPPG"].values
                ecg = sig_df["ECG"].values
                if len(timestamps) < 2:
                    continue

                dt = np.median(np.diff(timestamps))
                if dt <= 0:
                    continue
                fs = 1.0 / dt

                window_samples = int(window_sec * fs)
                step_samples = int(step_sec * fs)
                if window_samples > len(rppg) or window_samples > len(ecg):
                    continue

                hospital_pid = hospital_lookup.get(lab_pid, f"m{mirror_num}_lab{lab_pid}")

                patient_windows = 0
                start = 0
                while start + window_samples <= len(rppg):
                    win_start = start
                    win_end = start + window_samples
                    rppg_win = rppg[win_start:win_end].copy()
                    ecg_win = ecg[win_start:win_end].copy()
                    start += step_samples

                    if np.isnan(rppg_win).any() or np.isnan(ecg_win).any():
                        continue

                    rppg_win = resample(rppg_win, target_length)
                    ecg_win = resample(ecg_win, target_length)

                    if np.isnan(rppg_win).any() or np.isnan(ecg_win).any():
                        continue
                    if rppg_win.std() < 1e-6 or ecg_win.std() < 1e-6:
                        continue
                    if (np.abs(rppg_win) < 1e-8).mean() > 0.95:
                        continue
                    if (np.abs(ecg_win) < 1e-8).mean() > 0.95:
                        continue

                    rppg_win = _zscore(rppg_win).astype(np.float32)
                    ecg_win = _zscore(ecg_win).astype(np.float32)

                    fs_target = target_length / window_sec
                    rppg_snr = compute_snr_db(rppg_win, fs_target, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)
                    ecg_sqi = ecg_sqi_autocorr(ecg_win, fs_target)

                    if self.signal_type == "ecg":
                        sig = ecg_win
                        quality_value = float(ecg_sqi)
                    else:
                        sig = rppg_win
                        quality_value = float(rppg_snr)

                    self.samples.append(sig[None, :])
                    self.hospital_pids.append(hospital_pid)
                    self.source_records.append(
                        {
                            "dataset_index": len(self.samples) - 1,
                            "mirror": mirror_name,
                            "file_path": fpath,
                            "file_name": fname,
                            "lab_patient_id": int(lab_pid),
                            "hospital_patient_id": str(hospital_pid),
                            "window_start_index": int(win_start),
                            "window_end_index": int(win_end),
                            "window_start_time": float(timestamps[win_start]),
                            "window_end_time": float(timestamps[win_end - 1]),
                            "window_samples": int(window_samples),
                            "step_samples": int(step_samples),
                            "sampling_rate_hz": float(fs),
                            "window_sec": float(window_sec),
                            "target_length": int(target_length),
                        }
                    )

                    self.rppg_snr_db.append(float(rppg_snr))
                    self.ecg_autocorr_sqi.append(float(ecg_sqi))
                    self.raw_quality.append(quality_value)

                    patient_windows += 1
                    if max_windows_per_patient is not None and patient_windows >= max_windows_per_patient:
                        break

        if not self.samples:
            raise RuntimeError(f"No valid windows for signal_type={self.signal_type}.")

        self.quality_score = _rank01(self.raw_quality).astype(np.float32).tolist()

        print(
            f"[Exp3 Single Dataset:{self.signal_type}] Loaded {len(self.samples)} windows from {patient_count} patients. "
            f"rPPG SNR range {np.min(self.rppg_snr_db):.2f}..{np.max(self.rppg_snr_db):.2f} dB, "
            f"ECG autocorr SQI range {np.min(self.ecg_autocorr_sqi):.3f}..{np.max(self.ecg_autocorr_sqi):.3f}, "
            f"quality score range {np.min(self.quality_score):.3f}..{np.max(self.quality_score):.3f}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sig = self.samples[idx]
        return (
            torch.from_numpy(sig),
            torch.tensor(self.quality_score[idx], dtype=torch.float32),
        )

    def get_source_record(self, idx):
        return self.source_records[idx]


def build_single_signal_dataloaders(
    root_dir,
    signal_type,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=256,
    data_source="sqi",
    max_windows_per_patient=None,
    max_patients=None,
):
    dataset = SingleSignalMaskedReconDataset(
        root_dir,
        signal_type=signal_type,
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
        raise RuntimeError("No valid samples after dataset loading.")

    n_val_pids = max(1, int(len(unique_pids) * val_ratio))
    n_val_pids = min(n_val_pids, len(unique_pids))

    val_pids = set(unique_pids[:n_val_pids])
    train_pids = set(unique_pids[n_val_pids:])

    train_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in train_pids]
    val_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in val_pids]

    # If capped data produces an empty side, fallback to a sample-level split.
    if len(train_indices) == 0 or len(val_indices) == 0:
        all_indices = np.arange(len(dataset))
        rng.shuffle(all_indices)
        n_val = max(1, int(len(all_indices) * val_ratio))
        n_val = min(n_val, len(all_indices) - 1) if len(all_indices) > 1 else 1
        val_indices = all_indices[:n_val].tolist()
        train_indices = all_indices[n_val:].tolist()
        if len(train_indices) == 0:
            train_indices = val_indices

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

    train_quality = np.array([dataset.quality_score[i] for i in train_indices])
    val_quality = np.array([dataset.quality_score[i] for i in val_indices])
    print(
        f"[Exp3 Single DataLoader:{signal_type}] train={len(train_indices)} samples, "
        f"val={len(val_indices)} samples | "
        f"quality mean train={train_quality.mean():.3f}, val={val_quality.mean():.3f}"
    )

    return train_loader, val_loader
