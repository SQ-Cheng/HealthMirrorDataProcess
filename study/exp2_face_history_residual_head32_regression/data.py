"""Reuse current ten-task labels and build fold-safe prior-history features."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from study.exp2_face_history_head32_regression import history_data
from study.exp2_face_history_head32_regression.history_data import HistoryFeatureStore
from study.exp2_face_history_head32_regression.scaling import fit_robust_target_scaler
from study.exp2_face_history_head32_regression.source_data import TARGET_ANALYTES, _load_lab_data, _sha256
from study.exp2_face_pretrained_head32_regression.config import ALL_REGRESSION_TARGETS, SCORE_DEFINITIONS


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
TASKS = ALL_REGRESSION_TARGETS


def validate_patient_split(records):
    if records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any():
        raise ValueError("Duplicate video or patient leakage in outer split")
    if set(records.split) != {"train", "val", "test"}:
        raise ValueError("All three outer splits are required")


def crossfit_splits(records, seed, folds=3):
    """OOF patients and inner early-stopping patients all come from outer train."""
    training = records.loc[records.split.eq("train")].reset_index(drop=True)
    bins = pd.qcut(training.raw_value, q=5, duplicates="drop", labels=False).fillna(0).astype(int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    coverage = np.zeros(len(training), dtype=int)
    result = []
    for fold, (development, held_out) in enumerate(splitter.split(training, bins, training.hospital_id)):
        inner = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed+fold+1)
        inner_train, inner_val = next(inner.split(training.iloc[development], groups=training.iloc[development].hospital_id))
        fit, validation = development[inner_train], development[inner_val]
        frame = pd.concat([training.iloc[fit].assign(split="train"),
                           training.iloc[validation].assign(split="val"),
                           training.iloc[held_out].assign(split="test")], ignore_index=True)
        validate_patient_split(frame)
        coverage[held_out] += 1
        result.append(frame)
    if not np.all(coverage == 1):
        raise AssertionError("Each outer-train video must have exactly one OOF prediction")
    return result


def rescale_history(store, fold_scaler, raw_history):
    features = store.features.copy()
    raw_history = np.asarray(raw_history, dtype=np.float64)
    if len(raw_history) != len(features) or not np.isfinite(raw_history).all():
        raise ValueError("Raw history rows must align with the compact feature store")
    features[:, 0] = fold_scaler.transform(raw_history).astype(np.float32)
    return HistoryFeatureStore(store.video_ids, store.offsets, features, store.max_length)


@dataclass(frozen=True)
class ResidualScaler:
    """Do not center residuals; retain the original training-label loss scale."""
    target: str
    unit: str
    iqr: float
    lab_scaler: dict

    def transform(self, values):
        return np.asarray(values, dtype=np.float64) / self.iqr

    def inverse_transform(self, values):
        return np.asarray(values, dtype=np.float64) * self.iqr

    def to_dict(self):
        return {"target": self.target, "unit": self.unit, "median": 0., "iqr": self.iqr,
                "method": "zero-centered residual divided by outer-train raw-label IQR",
                "lab_target_scaler": self.lab_scaler}


def residual_records(records, baseline, scaler):
    if baseline.video_id.duplicated().any():
        raise ValueError("History predictions must be unique per video")
    merged = records.merge(baseline[["hospital_id", "video_id", "history_pred_raw", "prediction_source"]],
                           on=["hospital_id", "video_id"], how="left", validate="one_to_one")
    if not np.isfinite(merged.history_pred_raw).all():
        raise ValueError("Missing/nonfinite history baseline")
    if not merged.loc[merged.split.eq("train"), "prediction_source"].eq("patient_oof").all():
        raise ValueError("Training residuals must use patient-OOF history predictions")
    merged["original_raw_value"] = merged.raw_value
    merged["original_score_threshold"] = merged.score_threshold
    merged["raw_value"] = merged.original_raw_value - merged.history_pred_raw
    merged["score_threshold"] = merged.original_score_threshold - merged.history_pred_raw
    merged["robust_scaled_raw_value"] = merged.raw_value / scaler.iqr
    return merged


def prepare(output, targets, lab_path, seed, smoke=False):
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    labs, _, quality = _load_lab_data(targets, source)
    lab_sha = _sha256(lab_path)
    current_timeseries = history_data.LAB_TIMESERIES_CACHE
    history_data.LAB_TIMESERIES_CACHE = str(source / "lab_timeseries.csv")
    summaries, counts, scalers = [], [], {}
    try:
        for target in targets:
            source_dir = REFERENCE / "source_data"
            if target in ("hematocrit_low", "egfr_low"):
                source_dir /= "direct_fields"
            saved_quality = json.loads((source_dir / "data_quality_report.json").read_text())
            fingerprint_quality = saved_quality
            if "source_fingerprints" not in fingerprint_quality:
                fingerprint_quality = json.loads((REFERENCE / "source_data/data_quality_report.json").read_text())
            fingerprint = fingerprint_quality["source_fingerprints"]["merged_lab_tests"]
            protocol_name = "direct_field_extension_protocol.json" if target in ("hematocrit_low", "egfr_low") else "main_protocol.json"
            protocol = json.loads((REFERENCE / protocol_name).read_text())
            if (protocol["lab_table_sha256"] != lab_sha
                    or fingerprint["size_bytes"] != Path(lab_path).stat().st_size
                    or fingerprint["mtime_ns"] != Path(lab_path).stat().st_mtime_ns):
                raise RuntimeError("Reference labels are stale; rebuild the main ten-task dataset first")
            records = pd.read_csv(REFERENCE / f"task_records/{target}.csv", dtype={"hospital_id": str})
            validate_patient_split(records)
            matched = records.merge(labs.loc[labs.analyte.eq(TARGET_ANALYTES[target])],
                                   left_on=["hospital_id", "label_time_unix"],
                                   right_on=["hospital_id", "timestamp_unix"], how="left", validate="many_to_one")
            if not np.allclose(matched.raw_value, matched.value, rtol=0, atol=1e-10):
                raise ValueError(f"Current source labels differ from main experiment: {target}")
            if smoke:
                parts = []
                for split, count in (("train", 36), ("val", 6), ("test", 6)):
                    pool = records.loc[records.split.eq(split)].drop_duplicates("hospital_id")
                    parts.append(pool.sample(n=min(count, len(pool)), random_state=seed))
                records = pd.concat(parts, ignore_index=True)
            scaler = fit_robust_target_scaler(target, records, SCORE_DEFINITIONS[target]["unit"])
            records["robust_scaled_raw_value"] = scaler.transform(records.raw_value)
            scalers[target] = scaler.to_dict()
            path = output / "task_records"
            path.mkdir(exist_ok=True)
            records.to_csv(path / f"{target}.csv", index=False)
            base = records[["source_sample_id", "label_time_unix"]].rename(columns={
                "source_sample_id": "sample_id", "label_time_unix": f"{TARGET_ANALYTES[target]}_lab_time_unix"})
            summary = history_data.build_history_artifacts(target, records, base, output / "history_records", scaler)
            summaries.append(summary)
            for split, group in records.groupby("split"):
                counts.append({"target": target, "split": split, "videos": len(group),
                               "patients": group.hospital_id.nunique(), "clinical_events": group.clinical_event_id.nunique()})
            print(f"[prepare] task={target} videos={len(records)} history_coverage={summary.has_history.mean():.3f}", flush=True)
    finally:
        history_data.LAB_TIMESERIES_CACHE = current_timeseries
    pd.concat(summaries, ignore_index=True).to_csv(output / "history_coverage.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "dataset_counts.csv", index=False)
    (output / "target_scalers.json").write_text(json.dumps(scalers, indent=2)+"\n")
    (source / "source_policy.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2)+"\n")
    return lab_sha
