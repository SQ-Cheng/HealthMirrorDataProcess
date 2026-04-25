"""Experiment 03 rPPG-only visualization entrypoint."""

from single_recon_visualize import run_visualization


if __name__ == "__main__":
    run_visualization(signal_type="rppg", exp_name="exp3_rppg")
