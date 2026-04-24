"""rPPG-only TCN dataloader wrapper for Exp3 split."""

import os
import sys


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(CUR_DIR)
COMMON_DIR = os.path.join(TRAIN_DIR, "exp3_common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from single_recon_dataloader import build_single_signal_dataloaders


def build_exp3_rppg_tcn_dataloaders(root_dir, **kwargs):
    return build_single_signal_dataloaders(root_dir, signal_type="rppg", **kwargs)
