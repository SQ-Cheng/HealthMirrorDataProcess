"""Run independent weight-decay and BatchNorm-statistics ablations."""

import json

from .config import WEIGHT_DECAY
from .plot_regularization_comparison import plot_comparison
from .plot_patient_diverse_comparison import plot_comparison as plot_pair
from .run_ablations import ABLATION_DIR, _prepare_sources, _run_variant
from .run_patient_diverse_schedule_ablation import STAGE_CONFIG, VARIANT as SCHEDULE_VARIANT


REFERENCE = ABLATION_DIR / SCHEDULE_VARIANT
VARIANTS = (
    ("patient_diverse_schedule_30_40_wd1e2", 1e-2, False),
    ("patient_diverse_schedule_30_40_bn_fixed", WEIGHT_DECAY, True),
)


def main():
    records_paths, scalers, index_path, source_hashes = _prepare_sources()
    if not (REFERENCE / "COMPLETE").is_file():
        raise RuntimeError(f"Reference ablation is incomplete: {REFERENCE}")
    with open(REFERENCE / "experiment_manifest.json", encoding="utf-8") as handle:
        reference = json.load(handle)
    expected = {
        "architecture": "efficientnet_b0",
        "training_protocol": "two_stage_full",
        "train_batch_policy": "patient_diverse",
        **STAGE_CONFIG,
    }
    if (reference["baseline_source_sha256"] != source_hashes
            or any(reference.get(key) != value for key, value in expected.items())
            or reference.get("weight_decay", WEIGHT_DECAY) != WEIGHT_DECAY
            or reference.get("batchnorm_running_stats_frozen", False)):
        raise RuntimeError("Reference data or training configuration differs")
    for name, _, _ in VARIANTS:
        if (ABLATION_DIR / name).exists():
            raise FileExistsError(f"Ablation output already exists: {ABLATION_DIR / name}")

    for name, weight_decay, freeze_batchnorm_stats in VARIANTS:
        _run_variant(
            name, "efficientnet_b0", "two_stage_full", records_paths, scalers,
            index_path, source_hashes, train_batch_policy="patient_diverse",
            stage_config=STAGE_CONFIG, weight_decay=weight_decay,
            freeze_batchnorm_stats=freeze_batchnorm_stats,
        )
        plot_pair(
            REFERENCE, ABLATION_DIR / name,
            reference_label="30/40 control",
            candidate_label="Weight decay 1e-2" if not freeze_batchnorm_stats else "Fixed BN statistics",
            figure_prefix="control",
        )
    plot_comparison(REFERENCE, ABLATION_DIR, VARIANTS)
    print("[regularization-ablations-complete]", flush=True)


if __name__ == "__main__":
    main()
