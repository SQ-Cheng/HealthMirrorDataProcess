"""rPPG-only dataloader wrapper for Exp3 split."""

import os
import sys


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(CUR_DIR)
COMMON_DIR = os.path.join(TRAIN_DIR, "exp3_common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from single_recon_dataloader import build_single_signal_dataloaders


def build_exp3_rppg_dataloaders(
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
):
    return build_single_signal_dataloaders(
        root_dir,
        signal_type="rppg",
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        window_sec=window_sec,
        step_sec=step_sec,
        target_length=target_length,
        data_source=data_source,
        max_windows_per_patient=max_windows_per_patient,
        max_patients=max_patients,
    )
