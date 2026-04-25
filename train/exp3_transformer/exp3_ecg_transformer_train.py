"""Experiment 03 ECG-only Transformer entrypoint (single-window masked reconstruction)."""

from single_recon_train_transformer import run_experiment


if __name__ == "__main__":
    run_experiment(signal_type="ecg", exp_name="exp3_ecg_transformer")
