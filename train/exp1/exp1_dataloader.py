"""
Experiment 01 Dataloader
========================
Loads ECG and rPPG signals from mirrorx_auto_cleaned directories,
applies sliding window augmentation (3-sec window, 1-sec step),
resamples to 1024 points, and Z-score normalizes.

Label source: cleaned_patient_info.csv (SBP = High_Blood_Pressure, DBP = Low_Blood_Pressure)

BP normalization: [40, 180] mmHg -> [0, 1]
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample

# BP normalization range
BP_MIN = 40.0
BP_MAX = 180.0


def normalize_bp(value):
    """Map BP value from [40, 180] mmHg to [0, 1]."""
    return (value - BP_MIN) / (BP_MAX - BP_MIN)


def denormalize_bp(value):
    """Map normalized BP value from [0, 1] back to mmHg."""
    return value * (BP_MAX - BP_MIN) + BP_MIN


class BPDataset(Dataset):
    """
    Dataset for blood pressure estimation from ECG and rPPG signals.

    Each sample is a (ecg, rppg, [sbp, dbp]) tuple where ecg and rppg
    are 1024-point Z-score normalized signal windows.
    """

    def __init__(self, root_dir, window_sec=3.0, step_sec=1.0, target_length=1024):
        """
        Args:
            root_dir:       Workspace root containing mirrorx_auto_cleaned dirs.
            window_sec:     Sliding window length in seconds.
            step_sec:       Sliding window step in seconds.
            target_length:  Number of sample points per window after resampling.
        """
        self.target_length = target_length
        self.samples = []          # list of (ecg_window, rppg_window, bp)
        self.hospital_pids = []    # Hospital_Patient_ID for each sample (for patient-level split)
        self.lab_pids = []         # Lab_Patient_ID for each sample (for debug reporting)
        self.mirror_nums = []      # mirror number (X from mirrorX_auto_cleaned) per sample
        self._nan_report = []      # list of (source, detail) for NaN events

        # Discover all mirrorx_auto_cleaned directories
        mirror_dirs = sorted(glob.glob(os.path.join(root_dir, "mirror*_auto_cleaned")))
        if not mirror_dirs:
            raise FileNotFoundError(f"No mirror*_auto_cleaned directories found in {root_dir}")

        for mirror_dir in mirror_dirs:
            # Extract the mirror number from directory name, e.g. mirror2_auto_cleaned -> 2
            dir_name = os.path.basename(mirror_dir)
            try:
                mirror_num = int(''.join(filter(str.isdigit, dir_name.split('_')[0])))
            except (ValueError, IndexError):
                mirror_num = -1

            info_path = os.path.join(mirror_dir, "cleaned_patient_info.csv")
            if not os.path.exists(info_path):
                print(f"[WARN] No cleaned_patient_info.csv in {mirror_dir}, skipping.")
                continue

            # Load patient info and build lookup {Lab_Patient_ID: (sbp, dbp, hospital_pid)}
            info_df = pd.read_csv(info_path)
            bp_lookup = {}
            for _, row in info_df.iterrows():
                pid = int(row["Lab_Patient_ID"])
                dbp = row["Low_Blood_Pressure"]
                sbp = row["High_Blood_Pressure"]
                # Skip patients without valid BP
                if dbp == -1 or sbp == -1:
                    continue
                hospital_pid = row["Hospital_Patient_ID"]
                bp_lookup[pid] = (float(sbp), float(dbp), hospital_pid, pid)

            # Find all patient signal files
            signal_files = sorted(glob.glob(os.path.join(mirror_dir, "patient_*.csv")))
            for fpath in signal_files:
                fname = os.path.basename(fpath)
                # Parse patient ID from filename: patient_000012_1.csv -> 12
                parts = fname.replace(".csv", "").split("_")
                try:
                    pid = int(parts[1])
                except (IndexError, ValueError):
                    continue

                if pid not in bp_lookup:
                    continue

                sbp, dbp, hospital_pid, lab_pid = bp_lookup[pid]

                # Load signal
                try:
                    sig_df = pd.read_csv(fpath)
                except Exception as e:
                    print(f"[WARN] Failed to read {fpath}: {e}")
                    continue

                if "Timestamp" not in sig_df.columns or "RPPG" not in sig_df.columns or "ECG" not in sig_df.columns:
                    continue

                timestamps = sig_df["Timestamp"].values
                rppg = sig_df["RPPG"].values
                ecg = sig_df["ECG"].values

                # NaN check on raw signal
                ecg_nan = int(np.isnan(ecg).sum())
                rppg_nan = int(np.isnan(rppg).sum())
                if ecg_nan > 0 or rppg_nan > 0:
                    msg = f"{fname}: raw ECG NaN={ecg_nan}, rPPG NaN={rppg_nan}"
                    self._nan_report.append(("raw", msg))
                    print(f"[NaN] {msg}")

                if len(timestamps) < 2:
                    continue

                # Estimate sampling rate from timestamps
                dt = np.median(np.diff(timestamps))
                if dt <= 0:
                    continue
                fs = 1.0 / dt

                window_samples = int(window_sec * fs)
                step_samples = int(step_sec * fs)

                if window_samples > len(ecg):
                    continue

                # Sliding window
                start = 0
                while start + window_samples <= len(ecg):
                    ecg_win = ecg[start: start + window_samples].copy()
                    rppg_win = rppg[start: start + window_samples].copy()

                    # Resample to target_length
                    ecg_win = resample(ecg_win, target_length)
                    rppg_win = resample(rppg_win, target_length)

                    # NaN check after resampling
                    if np.isnan(ecg_win).any() or np.isnan(rppg_win).any():
                        msg = (f"{fname} window[{start}:{start+window_samples}]: "
                               f"NaN after resample  ECG={int(np.isnan(ecg_win).sum())}  "
                               f"rPPG={int(np.isnan(rppg_win).sum())}")
                        self._nan_report.append(("resample", msg))
                        print(f"[NaN] {msg}")
                        start += step_samples
                        continue

                    # Z-score normalization per window
                    ecg_win = self._zscore(ecg_win)
                    rppg_win = self._zscore(rppg_win)

                    # NaN check after z-score
                    if np.isnan(ecg_win).any() or np.isnan(rppg_win).any():
                        msg = (f"{fname} window[{start}:{start+window_samples}]: "
                               f"NaN after zscore  ECG={int(np.isnan(ecg_win).sum())}  "
                               f"rPPG={int(np.isnan(rppg_win).sum())}")
                        self._nan_report.append(("zscore", msg))
                        print(f"[NaN] {msg}")
                        start += step_samples
                        continue

                    self.samples.append((
                        ecg_win.astype(np.float32),
                        rppg_win.astype(np.float32),
                        np.array([normalize_bp(sbp), normalize_bp(dbp)], dtype=np.float32),
                    ))
                    self.hospital_pids.append(hospital_pid)
                    self.lab_pids.append(lab_pid)
                    self.mirror_nums.append(mirror_num)

                    start += step_samples

        nan_count = len(self._nan_report)
        if nan_count == 0:
            print(f"[BPDataset] Loaded {len(self.samples)} windows from {len(mirror_dirs)} mirror dir(s). No NaN detected.")
        else:
            print(f"[BPDataset] Loaded {len(self.samples)} windows from {len(mirror_dirs)} mirror dir(s). "
                  f"NaN events: {nan_count} (raw={sum(1 for s,_ in self._nan_report if s=='raw')}, "
                  f"resample={sum(1 for s,_ in self._nan_report if s=='resample')}, "
                  f"zscore={sum(1 for s,_ in self._nan_report if s=='zscore')})")

    def nan_report(self):
        """Print a full NaN event report."""
        if not self._nan_report:
            print("[NaN Report] No NaN events detected.")
            return
        print(f"[NaN Report] {len(self._nan_report)} event(s):")
        for stage, detail in self._nan_report:
            print(f"  [{stage}] {detail}")

    @staticmethod
    def _zscore(x):
        std = x.std()
        if std < 1e-8:
            return x - x.mean()
        return (x - x.mean()) / std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg, rppg, bp = self.samples[idx]
        # Shape: (1, 1024) for Conv1d input
        ecg_tensor = torch.from_numpy(ecg).unsqueeze(0)
        rppg_tensor = torch.from_numpy(rppg).unsqueeze(0)
        bp_tensor = torch.from_numpy(bp)
        return ecg_tensor, rppg_tensor, bp_tensor


def build_dataloaders(root_dir, batch_size=32, val_ratio=0.2, seed=42,
                      window_sec=3.0, step_sec=1.0, target_length=1024,
                      debug=False):
    """
    Build train and validation DataLoaders with a patient-level split.

    Samples sharing the same Hospital_Patient_ID are kept exclusively in
    either the training set or the validation set, preventing data leakage.

    Returns:
        train_loader, val_loader
    """
    dataset = BPDataset(root_dir, window_sec=window_sec, step_sec=step_sec,
                        target_length=target_length)

    # ── Patient-level split ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    unique_pids = list(dict.fromkeys(dataset.hospital_pids))  # preserve order, deduplicate
    rng.shuffle(unique_pids)

    n_val_pids = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val_pids])
    train_pids = set(unique_pids[n_val_pids:])

    train_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in train_pids]
    val_indices   = [i for i, p in enumerate(dataset.hospital_pids) if p in val_pids]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set   = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"[DataLoader] Patient-level split — "
          f"train: {len(train_pids)} patients / {len(train_indices)} samples, "
          f"val: {len(val_pids)} patients / {len(val_indices)} samples")
    dataset.nan_report()

    # ── BP distribution report ───────────────────────────────────────────
    _report_bp_stats(dataset, train_indices, val_indices)

    # ── Debug: report and save Lab_Patient_ID split ──────────────────────
    if debug:
        _save_split_report(dataset, train_indices, val_indices, save_dir=root_dir)

    return train_loader, val_loader


def _report_bp_stats(dataset, train_indices, val_indices):
    """Print Mean and STDDEV of SBP and DBP (in mmHg) for train and val sets."""
    def extract_bp(indices):
        sbp = np.array([denormalize_bp(dataset.samples[i][2][0]) for i in indices], dtype=np.float32)
        dbp = np.array([denormalize_bp(dataset.samples[i][2][1]) for i in indices], dtype=np.float32)
        return sbp, dbp

    tr_sbp, tr_dbp = extract_bp(train_indices)
    va_sbp, va_dbp = extract_bp(val_indices)

    print(f"\n[BP Stats]  {'':10}  {'SBP Mean':>10}  {'SBP SD':>8}  {'DBP Mean':>10}  {'DBP SD':>8}")
    print(f"            {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"            {'Train':>10}  {tr_sbp.mean():>10.2f}  {tr_sbp.std():>8.2f}  {tr_dbp.mean():>10.2f}  {tr_dbp.std():>8.2f}")
    print(f"            {'Val':>10}  {va_sbp.mean():>10.2f}  {va_sbp.std():>8.2f}  {va_dbp.mean():>10.2f}  {va_dbp.std():>8.2f}")


def _save_split_report(dataset, train_indices, val_indices, save_dir):
    """Print and save the Lab_Patient_ID -> train/val assignment with mirror info.
    Each (lab_pid, mirror) combination is reported on its own row."""
    from collections import Counter, defaultdict

    # Count samples per (lab_pid, mirror, split)
    train_key_counts = Counter(
        (dataset.lab_pids[i], dataset.mirror_nums[i]) for i in train_indices
    )
    val_key_counts = Counter(
        (dataset.lab_pids[i], dataset.mirror_nums[i]) for i in val_indices
    )

    all_keys = sorted(set(train_key_counts) | set(val_key_counts))  # (pid, mirror)

    # Build CSV rows ordered by pid then mirror
    rows = []
    for pid, mirror in sorted(set((p, m) for p, m in all_keys)):
        if (pid, mirror) in train_key_counts:
            rows.append((pid, "train", train_key_counts[(pid, mirror)], mirror))
        if (pid, mirror) in val_key_counts:
            rows.append((pid, "val", val_key_counts[(pid, mirror)], mirror))

    lines = ["Lab_Patient_ID,Split,Samples,Mirror"] + \
            [f"{pid},{split},{samples},{mirror}" for pid, split, samples, mirror in rows]

    # Console summary
    
    #print(f"\n[Split Debug] Lab_Patient_ID assignment:")
    #print(f"  {'Lab_PID':>10}  {'Split':>5}  {'Samples':>7}  {'Mirror':>6}")
    #print(f"  {'-'*10}  {'-'*5}  {'-'*7}  {'-'*6}")
    #for pid, split, samples, mirror in rows:
    #    split_label = "train" if split == "train" else "  val"
    #    print(f"  {pid:>10}  {split_label:>5}  {samples:>7}  {mirror:>6}")
    
    # Save to file
    out_path = os.path.join(save_dir, "split_debug.csv")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Split Debug] Saved to {out_path}")


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_loader, val_loader = build_dataloaders(ROOT, batch_size=16, debug=True)
    for ecg, rppg, bp in train_loader:
        print(f"ECG: {ecg.shape}, RPPG: {rppg.shape}, BP: {bp.shape}")
        print(f"BP sample (normalized): {bp[0]}")
        from exp1_dataloader import denormalize_bp
        print(f"BP sample (mmHg):       {denormalize_bp(bp[0])}")
        break
