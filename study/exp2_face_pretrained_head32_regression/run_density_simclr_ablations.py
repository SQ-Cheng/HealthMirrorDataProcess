"""Two independent changes to the completed patient-diverse 30/40 control."""

import json

from .config import TARGETS
from .plot_density_simclr_comparison import plot_comparison as plot_three_way
from .plot_patient_diverse_comparison import plot_comparison
from .run_ablations import ABLATION_DIR, _prepare_sources, _run_variant
from .run_patient_diverse_schedule_ablation import STAGE_CONFIG


REFERENCE = ABLATION_DIR / "patient_diverse_schedule_30_40"
DENSITY = "patient_diverse_schedule_30_40_density_weighted"
SIMCLR = "patient_diverse_schedule_30_40_label_simclr"


def _validate_reference(source_hashes):
    if not (REFERENCE / "COMPLETE").is_file():
        raise RuntimeError(f"Control is incomplete: {REFERENCE}")
    with open(REFERENCE / "experiment_manifest.json", encoding="utf-8") as handle:
        control = json.load(handle)
    if (control["baseline_source_sha256"] != source_hashes
            or control["train_batch_policy"] != "patient_diverse"
            or control["training_protocol"] != "two_stage_full"
            or control["architecture"] != "efficientnet_b0"
            or control["targets"] != list(TARGETS)):
        raise RuntimeError("Control uses different data, model, or batch policy")
    for key, expected in STAGE_CONFIG.items():
        if control[key] != expected:
            raise RuntimeError(f"Control schedule differs at {key}")


def main():
    records_paths, scalers, index_path, source_hashes = _prepare_sources()
    _validate_reference(source_hashes)
    _run_variant(
        DENSITY, "efficientnet_b0", "two_stage_full",
        records_paths, scalers, index_path, source_hashes,
        train_batch_policy="patient_diverse", stage_config=STAGE_CONFIG,
        density_weighting=True,
    )
    plot_comparison(
        REFERENCE, ABLATION_DIR / DENSITY,
        reference_label="Patient-diverse 30/40",
        candidate_label="Inverse-density training loss",
        figure_prefix="density",
    )
    _run_variant(
        SIMCLR, "efficientnet_b0", "two_stage_full",
        records_paths, scalers, index_path, source_hashes,
        train_batch_policy="patient_diverse", stage_config=STAGE_CONFIG,
        simclr_pretraining=True,
    )
    plot_comparison(
        REFERENCE, ABLATION_DIR / SIMCLR,
        reference_label="Patient-diverse 30/40",
        candidate_label="Label-aware SimCLR initialization",
        figure_prefix="simclr",
    )
    plot_three_way(REFERENCE, ABLATION_DIR / DENSITY, ABLATION_DIR / SIMCLR)
    print("[density-simclr-ablations-complete]", flush=True)


if __name__ == "__main__":
    main()
