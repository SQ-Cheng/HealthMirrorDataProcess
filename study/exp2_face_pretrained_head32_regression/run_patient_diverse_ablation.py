"""Run the patient-diverse batch ablation with the baseline's data and settings."""

from .plot_patient_diverse_comparison import plot_comparison
from .run_ablations import ABLATION_DIR, BASE_DIR, _prepare_sources, _run_variant


VARIANT = "patient_diverse_batches"


def main():
    records_paths, scalers, index_path, source_hashes = _prepare_sources()
    _run_variant(
        VARIANT, "efficientnet_b0", "two_stage_full", records_paths, scalers,
        index_path, source_hashes, train_batch_policy="patient_diverse",
    )
    plot_comparison(BASE_DIR, ABLATION_DIR / VARIANT)
    print("[patient-diverse-ablation-complete]", flush=True)


if __name__ == "__main__":
    main()
