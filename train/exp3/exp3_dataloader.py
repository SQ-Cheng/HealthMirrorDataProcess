"""Experiment 03 dataloader: joint ECG+rPPG masked reconstruction with quality ranking."""

import glob
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, resample
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


def ecg_sqi_template_corr(ecg, fs):
    """Beat-template similarity SQI in [0,1]."""
    prominence = max(0.2, 0.25 * np.std(ecg))
    min_dist = max(1, int(fs * 0.25))
    peaks, _ = find_peaks(ecg, distance=min_dist, prominence=prominence)
    if len(peaks) < 4:
        return 0.0

    pre = max(1, int(fs * 0.20))
    post = max(1, int(fs * 0.40))
    beats = []
    for p in peaks:
        s = p - pre
        e = p + post
        if s < 0 or e > len(ecg):
            continue
        beats.append(ecg[s:e])

    if len(beats) < 3:
        return 0.0

    beats = np.stack(beats, axis=0)
    template = np.median(beats, axis=0)
    t_std = template.std()
    if t_std < 1e-8:
        return 0.0

    corrs = []
    for b in beats:
        b_std = b.std()
        if b_std < 1e-8:
            continue
        c = np.corrcoef(b, template)[0, 1]
        if np.isfinite(c):
            corrs.append(c)

    if not corrs:
        return 0.0

    c_mean = float(np.mean(corrs))
    return float(np.clip((c_mean + 1.0) * 0.5, 0.0, 1.0))


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


def ecg_sqi_morph_stability(ecg, n_splits=6):
    """Morphology stability from adjacent split correlation in [0,1]."""
    chunks = np.array_split(ecg, n_splits)
    if len(chunks) < 2:
        return 0.0

    corrs = []
    for i in range(len(chunks) - 1):
        a = chunks[i]
        b = chunks[i + 1]
        n = min(len(a), len(b))
        if n < 8:
            continue
        a = a[:n]
        b = b[:n]
        if a.std() < 1e-8 or b.std() < 1e-8:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if np.isfinite(c):
            corrs.append(c)

    if not corrs:
        return 0.0

    c_mean = float(np.mean(corrs))
    return float(np.clip((c_mean + 1.0) * 0.5, 0.0, 1.0))


def ecg_sqi_artifact_penalty(ecg):
    """Artifact-based SQI in [0,1]; higher is cleaner."""
    diff = np.abs(np.diff(ecg))
    flat_ratio = float((diff < 1e-3).mean()) if len(diff) else 1.0
    spike_ratio = float((np.abs(ecg) > 3.5).mean())
    jump_ratio = float((diff > 2.0).mean()) if len(diff) else 0.0

    penalty = 0.45 * flat_ratio + 0.30 * spike_ratio + 0.25 * jump_ratio
    score = 1.0 - penalty
    return float(np.clip(score, 0.0, 1.0))


def ecg_sqi_composite(ecg, fs):
    """Non-frequency ECG SQI composite from multiple time-domain methods."""
    s_template = ecg_sqi_template_corr(ecg, fs)
    s_ac = ecg_sqi_autocorr(ecg, fs)
    s_morph = ecg_sqi_morph_stability(ecg)
    s_art = ecg_sqi_artifact_penalty(ecg)

    # Emphasize beat consistency and periodicity; keep artifact and stability as supports.
    return float(0.40 * s_template + 0.30 * s_ac + 0.20 * s_art + 0.10 * s_morph)


class MaskedReconDataset(Dataset):
    """Per-window ECG+rPPG samples with quality score derived from ranked quality proxies."""

    def __init__(
        self,
        root_dir,
        window_sec=3.0,
        step_sec=1.0,
        target_length=256,
        max_windows_per_patient=None,
        max_patients=None,
    ):
        self.samples = []
        self.hospital_pids = []

        self.rppg_snr_db = []
        self.ecg_quality = []
        self.ecg_template_sqi = []
        self.ecg_autocorr_sqi = []
        self.ecg_morph_sqi = []
        self.ecg_artifact_sqi = []
        self.ecg_legacy_freq_snr = []

        self.clean_score = []

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
                    rppg_win = rppg[start:start + window_samples].copy()
                    ecg_win = ecg[start:start + window_samples].copy()
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
                    pair = np.stack([ecg_win, rppg_win], axis=0)

                    fs_target = target_length / window_sec
                    rppg_snr = compute_snr_db(rppg_win, fs_target, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)

                    s_template = ecg_sqi_template_corr(ecg_win, fs_target)
                    s_ac = ecg_sqi_autocorr(ecg_win, fs_target)
                    s_morph = ecg_sqi_morph_stability(ecg_win)
                    s_art = ecg_sqi_artifact_penalty(ecg_win)
                    s_ecg = ecg_sqi_composite(ecg_win, fs_target)

                    ecg_snr_legacy = compute_snr_db(ecg_win, fs_target, lo_hz=1.0, hi_hz=30.0, peak_width_hz=0.8)

                    self.samples.append(pair)
                    self.hospital_pids.append(hospital_pid)

                    self.rppg_snr_db.append(float(rppg_snr))
                    self.ecg_quality.append(float(s_ecg))
                    self.ecg_template_sqi.append(float(s_template))
                    self.ecg_autocorr_sqi.append(float(s_ac))
                    self.ecg_morph_sqi.append(float(s_morph))
                    self.ecg_artifact_sqi.append(float(s_art))
                    self.ecg_legacy_freq_snr.append(float(ecg_snr_legacy))

                    patient_windows += 1
                    if max_windows_per_patient is not None and patient_windows >= max_windows_per_patient:
                        break

        if not self.samples:
            raise RuntimeError("No valid ECG+rPPG windows for Experiment 03.")

        rppg_rank = _rank01(self.rppg_snr_db)
        ecg_rank = _rank01(self.ecg_quality)
        self.clean_score = (0.6 * rppg_rank + 0.4 * ecg_rank).astype(np.float32).tolist()

        print(
            f"[Exp3 Dataset] Loaded {len(self.samples)} ECG+rPPG windows from {patient_count} patients. "
            f"rPPG SNR range {np.min(self.rppg_snr_db):.2f}..{np.max(self.rppg_snr_db):.2f} dB, "
            f"ECG SQI range {np.min(self.ecg_quality):.3f}..{np.max(self.ecg_quality):.3f}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pair = self.samples[idx]
        return (
            torch.from_numpy(pair),
            torch.tensor(self.clean_score[idx], dtype=torch.float32),
            torch.tensor(self.ecg_quality[idx], dtype=torch.float32),
            torch.tensor(self.rppg_snr_db[idx], dtype=torch.float32),
        )


def build_masked_recon_dataloaders(
    root_dir,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    window_sec=3.0,
    step_sec=1.0,
    target_length=256,
    max_windows_per_patient=None,
    max_patients=None,
):
    dataset = MaskedReconDataset(
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
        raise RuntimeError("No valid samples for Experiment 03.")

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

    train_clean = np.array([dataset.clean_score[i] for i in train_indices])
    val_clean = np.array([dataset.clean_score[i] for i in val_indices])
    print(
        "[Exp3 DataLoader] Patient-level split - "
        f"train: {len(train_pids)} patients / {len(train_indices)} samples, "
        f"val: {len(val_pids)} patients / {len(val_indices)} samples | "
        f"clean-score mean train={train_clean.mean():.3f}, val={val_clean.mean():.3f}"
    )

    return train_loader, val_loader
