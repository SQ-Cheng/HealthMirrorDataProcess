"""Add two warmup epochs before each unchanged 30/40 cosine schedule."""

import json

from .plot_patient_diverse_comparison import plot_comparison
from .run_ablations import ABLATION_DIR, _prepare_sources, _run_variant
from .run_patient_diverse_schedule_ablation import STAGE_CONFIG


REFERENCE = ABLATION_DIR / "patient_diverse_schedule_30_40"
VARIANT = "patient_diverse_schedule_30_40_warmup2"
WARMUP_CONFIG = {
    **STAGE_CONFIG,
    "head_max_epochs": STAGE_CONFIG["head_max_epochs"] + 2,
    "finetune_max_epochs": STAGE_CONFIG["finetune_max_epochs"] + 2,
    "head_warmup_epochs": 2,
    "finetune_warmup_epochs": 2,
}


def main():
    records_paths, scalers, index_path, source_hashes = _prepare_sources()
    if not (REFERENCE / "COMPLETE").is_file():
        raise RuntimeError(f"Reference ablation is incomplete: {REFERENCE}")
    reference = json.loads((REFERENCE / "experiment_manifest.json").read_text())
    if (reference["baseline_source_sha256"] != source_hashes
            or reference["train_batch_policy"] != "patient_diverse"
            or reference["training_protocol"] != "two_stage_full"):
        raise RuntimeError("Reference data, batch policy, or protocol differs")
    for stage in ("head", "finetune"):
        for field in ("learning_rate", "min_learning_rate", "max_epochs", "patience"):
            expected = STAGE_CONFIG.get(
                f"{stage}_{field}",
                1e-6 if field == "min_learning_rate" else None,
            )
            if reference[f"{stage}_{field}"] != expected:
                raise RuntimeError(f"Reference {stage}_{field} differs")
    _run_variant(
        VARIANT, "efficientnet_b0", "two_stage_full", records_paths, scalers,
        index_path, source_hashes, train_batch_policy="patient_diverse",
        stage_config=WARMUP_CONFIG,
    )
    plot_comparison(
        REFERENCE, ABLATION_DIR / VARIANT,
        reference_label="Patient-diverse 30/40",
        candidate_label="Patient-diverse 2+30 / 2+40 warmup",
        figure_prefix="warmup2",
    )
    print("[patient-diverse-warmup-ablation-complete]", flush=True)


if __name__ == "__main__":
    main()
