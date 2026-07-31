"""Prepare aligned 20-frame task records for the Head32 last-stage experiment."""

import hashlib
import json
import os
import shutil

import pandas as pd

from .config import (
    ALIGNED_RECORDS_DIR,
    ALIGNED_REGRESSION_DIR,
    SEED,
    SHARED_INDEX_DIR,
)
from study.exp2_face_pretrained_head32_regression.data import add_patient_split
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex


REUSED_TARGETS = ("hemoglobin_low", "po2_low")
LACTATE_TARGET = "lactate_high"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_summary_row(template, records, excluded_count):
    row = template.copy()
    row["status"] = "ready"
    row["reason"] = ""
    row["frame_ineligible_videos"] = int(excluded_count)
    row["clean_videos"] = int(len(records))
    row["clean_patients"] = int(records["hospital_id"].nunique())
    row["positive_videos"] = int(records["binary_label"].eq(1).sum())
    row["negative_videos"] = int(records["binary_label"].eq(0).sum())
    row["neutral_boundary_videos"] = int(records["abnormal_score"].eq(0).sum())
    row["score_min"] = float(records["abnormal_score"].min())
    row["score_median"] = float(records["abnormal_score"].median())
    row["score_max"] = float(records["abnormal_score"].max())
    for split in ("train", "val", "test"):
        selected = records.loc[records["split"].eq(split)]
        row[f"{split}_patients"] = int(selected["hospital_id"].nunique())
        row[f"{split}_videos"] = int(len(selected))
        row[f"{split}_source_frames"] = int(len(selected) * 20)
    row["train_augmented_inputs"] = int(row["train_source_frames"] * 5)
    return row


def main():
    output_20 = os.path.join(ALIGNED_REGRESSION_DIR, "outputs", "20frame")
    output_all = os.path.join(ALIGNED_REGRESSION_DIR, "outputs", "allframes")
    input_dir = os.path.dirname(ALIGNED_RECORDS_DIR)
    frame_index = FrameOffsetIndex.load(
        os.path.join(SHARED_INDEX_DIR, "frame_offsets.npz")
    )

    shutil.rmtree(input_dir, ignore_errors=True)
    os.makedirs(ALIGNED_RECORDS_DIR, exist_ok=True)

    source_paths = {}
    for target in REUSED_TARGETS:
        source = os.path.join(output_20, "task_records", f"{target}.csv")
        destination = os.path.join(ALIGNED_RECORDS_DIR, f"{target}.csv")
        shutil.copy2(source, destination)
        source_paths[target] = source

    lactate_source = os.path.join(
        output_all,
        "task_records",
        f"{LACTATE_TARGET}.csv",
    )
    lactate = pd.read_csv(
        lactate_source,
        dtype={"hospital_id": str, "video_id": str},
    )
    indexed_videos = set(frame_index.video_lookup)
    eligible = lactate.loc[lactate["video_id"].isin(indexed_videos)].copy()
    excluded = lactate.loc[~lactate["video_id"].isin(indexed_videos)].copy()
    eligible = eligible.drop(columns="split", errors="ignore")
    eligible, reason, audit_rows, pair_rows, selection = add_patient_split(
        eligible,
        LACTATE_TARGET,
        seed=SEED,
    )
    if eligible is None:
        raise RuntimeError(f"Could not assign Lactate split: {reason}")
    eligible.to_csv(
        os.path.join(ALIGNED_RECORDS_DIR, f"{LACTATE_TARGET}.csv"),
        index=False,
    )
    source_paths[LACTATE_TARGET] = lactate_source

    summary_20 = pd.read_csv(os.path.join(output_20, "task_summary.csv"))
    summary_all = pd.read_csv(os.path.join(output_all, "task_summary.csv"))
    summaries = summary_20.loc[
        summary_20["target"].isin(REUSED_TARGETS)
    ].copy()
    summaries["frame_ineligible_videos"] = 0
    lactate_template = summary_all.loc[
        summary_all["target"].eq(LACTATE_TARGET)
    ].iloc[0]
    lactate_summary = _task_summary_row(
        lactate_template,
        eligible,
        len(excluded),
    )
    summaries = pd.concat(
        [summaries, pd.DataFrame([lactate_summary])],
        ignore_index=True,
    )
    summaries.to_csv(
        os.path.join(input_dir, "task_summary.csv"),
        index=False,
    )

    audit_20 = pd.read_csv(
        os.path.join(output_20, "split_distribution_audit.csv")
    )
    pairwise_20 = pd.read_csv(
        os.path.join(output_20, "split_distribution_pairwise.csv")
    )
    pd.concat(
        [audit_20, pd.DataFrame(audit_rows)],
        ignore_index=True,
    ).to_csv(
        os.path.join(input_dir, "split_distribution_audit.csv"),
        index=False,
    )
    pd.concat(
        [pairwise_20, pd.DataFrame(pair_rows)],
        ignore_index=True,
    ).to_csv(
        os.path.join(input_dir, "split_distribution_pairwise.csv"),
        index=False,
    )

    with open(
        os.path.join(output_20, "split_assignment_manifest.json"),
        encoding="utf-8",
    ) as handle:
        split_manifest = json.load(handle)
    split_manifest["target_results"].append(selection)
    split_manifest["lactate_20frame_eligibility"] = {
        "source_records": os.path.abspath(lactate_source),
        "source_videos": int(len(lactate)),
        "eligible_videos": int(len(eligible)),
        "excluded_videos": int(len(excluded)),
        "excluded_video_ids": sorted(excluded["video_id"].astype(str)),
        "policy": "exclude videos absent from the validated shared 20-frame index",
    }
    with open(
        os.path.join(input_dir, "split_assignment_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(split_manifest, handle, indent=2)

    pd.DataFrame(
        columns=(
            "target",
            "video_id",
            "hospital_id",
            "positive_event_count",
            "negative_event_count",
            "event_count",
            "action",
        )
    ).to_csv(
        os.path.join(input_dir, "conflicting_videos.csv"),
        index=False,
    )
    with open(
        os.path.join(input_dir, "input_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "seed": int(SEED),
                "frame_index": os.path.abspath(SHARED_INDEX_DIR),
                "head32_exact_reuse_targets": list(REUSED_TARGETS),
                "lactate_split_policy": (
                    "filter against the shared 20-frame index, then apply the "
                    "same patient-disjoint distribution-balanced split"
                ),
                "source_sha256": {
                    target: _sha256(path)
                    for target, path in source_paths.items()
                },
            },
            handle,
            indent=2,
        )
    print(
        "Prepared Head32 last-stage inputs: "
        f"Hb={len(pd.read_csv(os.path.join(ALIGNED_RECORDS_DIR, 'hemoglobin_low.csv')))} "
        f"PO2={len(pd.read_csv(os.path.join(ALIGNED_RECORDS_DIR, 'po2_low.csv')))} "
        f"Lactate={len(eligible)} excluded_lactate={len(excluded)}"
    )


if __name__ == "__main__":
    main()
