"""Experiment 03 ECG-only TCN entrypoint (single-window masked reconstruction)."""

from single_recon_train_tcn import run_experiment


if __name__ == "__main__":
    run_experiment(signal_type="ecg", exp_name="exp3_ecg_tcn")
