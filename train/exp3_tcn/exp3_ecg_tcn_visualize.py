"""Experiment 03 ECG-only TCN visualization entrypoint."""

from single_recon_visualize_tcn import run_visualization


if __name__ == "__main__":
    run_visualization(signal_type="ecg", exp_name="exp3_ecg_tcn")
