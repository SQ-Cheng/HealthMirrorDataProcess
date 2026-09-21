"""Prepare the shared frame index and the replacement bilirubin task."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from study.exp2_face_history_head32_regression import config as base_config
from study.exp2_face_history_head32_regression import data as base_data
from study.exp2_face_history_head32_regression import history_data
from study.exp2_face_history_head32_regression import source_data
from study.exp2_face_history_head32_regression.frame_index import (
    build_or_reuse_frame_index,
)
from study.exp2_face_history_head32_regression.scaling import (
    fit_robust_target_scaler,
)

from .engine import PREPARED_DIR, REFERENCE_DIR


TARGET = "total_bilirubin_high"
ANALYTE = "total_bilirubin"
DEFINITION = {
    "value_column": "total_bilirubin_value",
    "direction": "high",
    "threshold": 21.0,
    "scale": 10.0,
    "unit": "umol/L",
}


def _register_target():
    source_data.TARGET_ANALYTES[TARGET] = ANALYTE
    source_data.SCORE_DEFINITIONS[TARGET] = DEFINITION
    base_data.SCORE_DEFINITIONS[TARGET] = DEFINITION
    history_data.TARGET_ANALYTES[TARGET] = ANALYTE


def prepare():
    _register_target()
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = PREPARED_DIR / "source_data"
    base_manifest, video_summary, quality = source_data.build_raw_video_source(
        str(source_dir), (TARGET,)
    )

    reference_videos = pd.read_csv(
        REFERENCE_DIR / "source_data/video_summary.csv",
        dtype={"video_id": str},
    )
    index_records = pd.concat(
        [
            reference_videos[["video_id", "mirror", "lab_patient_id"]],
            video_summary[["video_id", "mirror", "lab_patient_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates("video_id").sort_values("video_id").reset_index(drop=True)
    frame_index = build_or_reuse_frame_index(
        index_records, PREPARED_DIR / "20frame_index", "20frame"
    )
    frame_manifest = json.loads(
        (PREPARED_DIR / "20frame_index/index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    usable = set(frame_index.video_lookup)
    base_manifest = base_manifest[base_manifest.video_id.astype(str).isin(usable)].copy()
    video_summary = video_summary[video_summary.video_id.astype(str).isin(usable)].copy()

    records, conflicts, summary = base_data.build_task_records(
        base_manifest, video_summary, TARGET
    )
    records, reason, distribution, pairwise, selection = base_data.add_patient_split(
        records, TARGET, base_config.SEED
    )
    if records is None:
        raise RuntimeError(f"Could not construct bilirubin split: {reason}")
    scaler = fit_robust_target_scaler(TARGET, records, DEFINITION["unit"])
    records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
    task_dir = PREPARED_DIR / "task_records"
    task_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(task_dir / f"{TARGET}.csv", index=False)
    pd.DataFrame(conflicts).to_csv(PREPARED_DIR / "bilirubin_conflicting_videos.csv", index=False)
    pd.DataFrame(distribution).to_csv(PREPARED_DIR / "bilirubin_split_distribution.csv", index=False)
    pd.DataFrame(pairwise).to_csv(PREPARED_DIR / "bilirubin_split_pairwise.csv", index=False)

    history_data.LAB_TIMESERIES_CACHE = str(source_dir / "lab_timeseries.csv")
    history_dir = PREPARED_DIR / "history_records"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_summary = history_data.build_history_artifacts(
        TARGET, records, base_manifest, str(history_dir), scaler
    )
    manifest = {
        "schema_version": 1,
        "target": TARGET,
        "definition": DEFINITION,
        "source_policy": quality["analyte_source_policies"][ANALYTE],
        "video_match_policy": quality["video_match_policy"],
        "frame_policy": frame_manifest["frame_policy"],
        "split": selection,
        "counts": {
            "videos": len(records),
            "patients": records.hospital_id.nunique(),
            "split_videos": records.groupby("split").size().to_dict(),
            "split_patients": records.groupby("split").hospital_id.nunique().to_dict(),
            "positive_videos": int(records.binary_label.eq(1).sum()),
            "negative_videos": int(records.binary_label.eq(0).sum()),
            "videos_with_history": int(history_summary.has_history.sum()),
        },
        "scaler_for_history_value_feature_only": scaler.to_dict(),
    }
    (PREPARED_DIR / "bilirubin_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[prepare-complete] target={TARGET} videos={len(records)} "
        f"patients={records.hospital_id.nunique()} split="
        f"{records.groupby('split').size().to_dict()} index_videos="
        f"{len(frame_index.video_ids)}",
        flush=True,
    )
    return records


if __name__ == "__main__":
    prepare()
