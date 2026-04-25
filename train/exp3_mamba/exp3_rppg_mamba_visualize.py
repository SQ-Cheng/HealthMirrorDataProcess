"""Experiment 03 rPPG-only Mamba visualization entrypoint."""

from single_recon_visualize_mamba import run_visualization


if __name__ == "__main__":
    run_visualization(signal_type="rppg", exp_name="exp3_rppg_mamba")
