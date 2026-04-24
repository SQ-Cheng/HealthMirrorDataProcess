"""ECG-only TCN model wrapper for Exp3 split."""

import os
import sys


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(CUR_DIR)
COMMON_DIR = os.path.join(TRAIN_DIR, "exp3_common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from single_recon_model_tcn import build_single_recon_tcn_model


def build_exp3_ecg_tcn_model(variant="light"):
    return build_single_recon_tcn_model(variant=variant)
