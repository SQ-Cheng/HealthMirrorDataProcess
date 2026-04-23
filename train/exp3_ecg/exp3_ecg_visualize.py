"""Experiment 03 ECG-only visualization entrypoint."""

import os
import sys


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(CUR_DIR)
COMMON_DIR = os.path.join(TRAIN_DIR, "exp3_common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from single_recon_visualize import run_visualization


if __name__ == "__main__":
    run_visualization(signal_type="ecg", exp_name="exp3_ecg")
