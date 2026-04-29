"""Experiment 03 ECG-only TCN entrypoint (single-window masked reconstruction)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp3_common.single_recon_train import run_experiment
from single_recon_model import build_single_recon_tcn_model

if __name__ == "__main__":
    run_experiment(
        signal_type="ecg",
        exp_name="exp3_ecg_tcn",
        model_builder_fn=build_single_recon_tcn_model,
        model_family="single_recon_tcn_v1",
        description_suffix=" (TCN)",
        plot_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
    )
