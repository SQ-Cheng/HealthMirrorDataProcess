"""Train the 30/40 control with distinct source frames per model batch."""

import json

from .config import WEIGHT_DECAY
from .plot_patient_diverse_comparison import plot_comparison
from .run_ablations import ABLATION_DIR, _prepare_sources, _run_variant
from .run_patient_diverse_schedule_ablation import STAGE_CONFIG, VARIANT as SCHEDULE_VARIANT


REFERENCE = ABLATION_DIR / SCHEDULE_VARIANT
VARIANT = "patient_diverse_schedule_30_40_interleaved_views"


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
    _run_variant(
        VARIANT, "efficientnet_b0", "two_stage_full", records_paths, scalers,
        index_path, source_hashes, train_batch_policy="interleaved_views",
        stage_config=STAGE_CONFIG,
    )
    plot_comparison(
        REFERENCE, ABLATION_DIR / VARIANT,
        reference_label="30/40 | grouped views",
        candidate_label="30/40 | interleaved views",
        figure_prefix="batch_diversity",
    )
    print("[interleaved-views-ablation-complete]", flush=True)


if __name__ == "__main__":
    main()
