"""Experiment 03 rPPG-only baseline visualization entrypoint."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp3_common.single_recon_visualize import run_visualization
from single_recon_model import build_single_recon_model

if __name__ == "__main__":
    run_visualization(
        signal_type="rppg",
        exp_name="exp3_rppg",
        model_builder_fn=build_single_recon_model,
        model_family="single_recon_v1",
        plot_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
    )
