"""Experiment 03 rPPG-only TCN visualization entrypoint."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp3_common.single_recon_visualize import run_visualization
from single_recon_model import build_single_recon_tcn_model

if __name__ == "__main__":
    run_visualization(
        signal_type="rppg",
        exp_name="exp3_rppg_tcn",
        model_builder_fn=build_single_recon_tcn_model,
        model_family="single_recon_tcn_v1",
        title_prefix=" TCN",
        plot_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
    )
