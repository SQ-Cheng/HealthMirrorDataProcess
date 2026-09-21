"""Prepare, validate, train on all visible GPUs, and plot Exp5 CASUS results."""

import argparse
import json

import pandas as pd

from study.exp4.frame_index import build_or_reuse_frame_index

from .build_dataset import add_patient_split, prepare_records, write_label_outputs
from .config import CACHE_DIR, CASUS_ANALYTES, CASUS_MAX_SCORE, OUTPUT_DIR, SEED
from .plot_results import plot_results
from .train import train


TRAINING_FILES = (
    "history.csv", "metrics.csv", "model.pt", "video_predictions.csv",
    "training_history.png", "run_manifest.json",
)


def prepare():
    records, frame_records, manifest = prepare_records(OUTPUT_DIR)
    frame_index, frame_manifest = build_or_reuse_frame_index(frame_records, CACHE_DIR)
    usable = set(frame_index.video_lookup)
    invalid = records[
        ~records.video_id.isin(usable) | ~records.pre_video_id.isin(usable)
    ].copy()
    if len(invalid):
        invalid["post_frame_usable"] = invalid.video_id.isin(usable)
        invalid["pre_frame_usable"] = invalid.pre_video_id.isin(usable)
        invalid.to_csv(OUTPUT_DIR / "frame_pair_exclusions.csv", index=False)
        records = records[
            records.video_id.isin(usable) & records.pre_video_id.isin(usable)
        ].copy()
        records = records.drop(columns="split")
        records, split_manifest = add_patient_split(records)
        records.to_csv(OUTPUT_DIR / "records.csv", index=False)
        write_label_outputs(records, OUTPUT_DIR)
        manifest["split"] = split_manifest
        print(
            f"[frames] excluded_pairs={len(invalid)} final_records={len(records)} "
            f"final_patients={records.hospital_id.nunique()} split_refit=true",
            flush=True,
        )
    else:
        (OUTPUT_DIR / "frame_pair_exclusions.csv").unlink(missing_ok=True)
    if records.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError("Patient leakage after frame validation")
    component_sum = sum(records[f"{name}_casus_points"] for name in CASUS_ANALYTES)
    if not records.casus_score.between(0, CASUS_MAX_SCORE).all() or not component_sum.equals(records.casus_score):
        raise AssertionError("Invalid partial-CASUS labels")
    manifest["counts"]["frame_excluded_pairs"] = len(invalid)
    manifest["counts"]["final_labelled_postoperative_videos"] = len(records)
    manifest["counts"]["final_labelled_patients"] = records.hospital_id.nunique()
    manifest["frame_index"] = frame_manifest["policy"]
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records, frame_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    records, frame_index = prepare()
    if args.prepare_only:
        return
    for name in TRAINING_FILES:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    train(records, frame_index, args.seed, OUTPUT_DIR)
    plot_results(OUTPUT_DIR)
    print("[experiment-complete] checkpoint, histories, predictions, and figures generated", flush=True)


if __name__ == "__main__":
    main()
