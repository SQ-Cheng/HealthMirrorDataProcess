"""Experiment 03 ECG-only Transformer entrypoint (single-window masked reconstruction)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp3_common.single_recon_train import run_experiment
from single_recon_model import build_single_recon_transformer_model

if __name__ == "__main__":
    run_experiment(
        signal_type="ecg",
        exp_name="exp3_ecg_transformer",
        model_builder_fn=build_single_recon_transformer_model,
        model_family="single_recon_transformer_v1",
        description_suffix=" (Transformer)",
        plot_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
    )
