"""Experiment 03 ECG-only Mamba visualization entrypoint."""

from single_recon_visualize_mamba import run_visualization


if __name__ == "__main__":
    run_visualization(signal_type="ecg", exp_name="exp3_ecg_mamba")
