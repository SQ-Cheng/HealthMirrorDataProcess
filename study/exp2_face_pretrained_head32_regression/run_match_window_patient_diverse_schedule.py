"""Apply the patient-diverse 30/40 schedule to a completed matching window."""

import argparse
import json

from .plot_patient_diverse_comparison import plot_comparison
from .run_ablations import ABLATION_DIR, _prepare_sources, _run_variant
from .run_patient_diverse_schedule_ablation import STAGE_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, choices=(6, 12), required=True)
    args = parser.parse_args()

    reference = ABLATION_DIR / f"lab_match_{args.hours}h"
    candidate_name = f"lab_match_{args.hours}h_patient_diverse_schedule_30_40"
    candidate = ABLATION_DIR / candidate_name
    if not (reference / "COMPLETE").is_file():
        raise RuntimeError(f"Matching-window reference is incomplete: {reference}")
    with open(reference / "experiment_manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (manifest["lab_match_max_delta_hours"] != args.hours
            or manifest["result_variant"] != "20frame"):
        raise RuntimeError(f"Unexpected source protocol: {reference}")

    records_paths, scalers, index_path, source_hashes = _prepare_sources(reference)
    _run_variant(
        candidate_name, "efficientnet_b0", "two_stage_full", records_paths,
        scalers, index_path, source_hashes, train_batch_policy="patient_diverse",
        stage_config=STAGE_CONFIG, baseline_dir=reference,
    )
    plot_comparison(
        reference, candidate,
        reference_label=f"{args.hours} h / original batches and schedule",
        candidate_label=f"{args.hours} h / patient-diverse 30/40",
        figure_prefix="schedule",
    )
    print(f"[window-schedule-complete] hours={args.hours} output={candidate}", flush=True)


if __name__ == "__main__":
    main()
