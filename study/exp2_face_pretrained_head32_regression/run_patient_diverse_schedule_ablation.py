"""Run the shorter two-stage schedule on the patient-diverse batch protocol."""

import json

from .plot_patient_diverse_comparison import plot_comparison
from .run_ablations import ABLATION_DIR, _prepare_sources, _run_variant


REFERENCE = ABLATION_DIR / "patient_diverse_batches"
VARIANT = "patient_diverse_schedule_30_40"
STAGE_CONFIG = {
    "head_learning_rate": 1e-4,
    "head_max_epochs": 30,
    "head_patience": 8,
    "finetune_learning_rate": 3e-6,
    "finetune_min_learning_rate": 1e-7,
    "finetune_max_epochs": 40,
    "finetune_patience": 8,
}


def main():
    records_paths, scalers, index_path, source_hashes = _prepare_sources()
    if not (REFERENCE / "COMPLETE").is_file():
        raise RuntimeError(f"Reference ablation is incomplete: {REFERENCE}")
    with open(REFERENCE / "experiment_manifest.json", encoding="utf-8") as handle:
        reference = json.load(handle)
    if (reference["train_batch_policy"] != "patient_diverse"
            or reference["training_protocol"] != "two_stage_full"
            or reference["baseline_source_sha256"] != source_hashes):
        raise RuntimeError("Reference data or training protocol differs")
    _run_variant(
        VARIANT, "efficientnet_b0", "two_stage_full", records_paths, scalers,
        index_path, source_hashes, train_batch_policy="patient_diverse",
        stage_config=STAGE_CONFIG,
    )
    plot_comparison(
        REFERENCE, ABLATION_DIR / VARIANT,
        reference_label="Patient-diverse / original schedule",
        candidate_label="Patient-diverse / shorter schedule",
        figure_prefix="schedule",
    )
    print("[patient-diverse-schedule-ablation-complete]", flush=True)


if __name__ == "__main__":
    main()
