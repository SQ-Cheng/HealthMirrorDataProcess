"""Experiment 03 rPPG-only TCN entrypoint (single-window masked reconstruction)."""

import os
import sys


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(CUR_DIR)
COMMON_DIR = os.path.join(TRAIN_DIR, "exp3_common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from single_recon_train_tcn import run_experiment


if __name__ == "__main__":
    run_experiment(signal_type="rppg", exp_name="exp3_rppg_tcn")
