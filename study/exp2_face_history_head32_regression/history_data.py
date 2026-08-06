"""Build and load leakage-safe same-episode prior-lab sequences."""

from dataclasses import dataclass
import json
import os

import numpy as np
import pandas as pd

from study.exp2_lab_multimodal.build_dataset import (
    _normalize_hospital_id,
    _parse_datetime_to_unix,
)
from study.exp2_lab_multimodal.config import LAB_CSV

from .config import (
    HISTORY_POLICY,
    HISTORY_TIME_SCALE_HOURS,
    LAB_TIMESERIES_CACHE,
    PO2_CANONICAL_ITEM_NAME,
    PO2_EXCLUDED_ITEM_NAMES,
)
from .source_data import TARGET_ANALYTES, _load_oxyhemoglobin_fraction


@dataclass(frozen=True)
class HistoryFeatureStore:
    video_ids: np.ndarray
    offsets: np.ndarray
    features: np.ndarray
    max_length: int

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                video_ids=data["video_ids"].astype(str),
                offsets=data["offsets"].astype(np.int64),
                features=data["features"].astype(np.float32),
                max_length=int(data["max_length"][0]),
            )

    def lookup(self):
        return {str(video_id): index for index, video_id in enumerate(self.video_ids)}


def _episode_windows():
    columns = ["首页病案号", "首页入院时间", "首页出院时间"]
    raw = pd.read_csv(
        LAB_CSV,
        dtype=str,
        keep_default_na=False,
        usecols=columns,
    )
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["admission_time_unix"] = _parse_datetime_to_unix(raw["首页入院时间"])
    raw["discharge_time_unix"] = _parse_datetime_to_unix(raw["首页出院时间"])
    valid = (
        raw["hospital_id"].ne("")
        & raw["admission_time_unix"].notna()
        & raw["discharge_time_unix"].notna()
        & raw["discharge_time_unix"].ge(raw["admission_time_unix"])
    )
    return (
        raw.loc[
            valid,
            ["hospital_id", "admission_time_unix", "discharge_time_unix"],
        ]
        .drop_duplicates()
        .sort_values(["hospital_id", "admission_time_unix", "discharge_time_unix"])
        .reset_index(drop=True)
    )


def build_history_artifacts(target, records, base_manifest, output_dir, scaler):
    """Write all strictly prior same-episode values and compact model features."""
    analyte = TARGET_ANALYTES[target]
    label_time_column = f"{analyte}_lab_time_unix"
    current_times = base_manifest[["sample_id", label_time_column]].rename(
        columns={"sample_id": "source_sample_id", label_time_column: "current_lab_time_unix"}
    )
    enriched = records.merge(
        current_times,
        on="source_sample_id",
        how="left",
        validate="one_to_one",
    )
    if enriched["current_lab_time_unix"].isna().any():
        raise ValueError(f"Missing current lab timestamps for {target}")

    if analyte == "oxyhemoglobin_fraction":
        labs, _ = _load_oxyhemoglobin_fraction()
    else:
        labs = pd.read_csv(
            LAB_TIMESERIES_CACHE,
            dtype={"hospital_id": str, "analyte": str},
        )
        labs = labs.loc[labs["analyte"].eq(analyte)].copy()
    if analyte == "po2":
        if "item_name" not in labs:
            raise ValueError("PO2 history cache lacks item_name")
        labs = labs.loc[
            labs["item_name"].eq(PO2_CANONICAL_ITEM_NAME)
        ].copy()
        if labs.empty:
            raise ValueError(
                f"No canonical PO2 rows found for {PO2_CANONICAL_ITEM_NAME}"
            )
    labs["timestamp_unix"] = pd.to_numeric(labs["timestamp_unix"], errors="coerce")
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs = labs.dropna(subset=["timestamp_unix", "value"])
    labs_by_patient = {
        hospital_id: group.sort_values(["timestamp_unix", "value"]).reset_index(drop=True)
        for hospital_id, group in labs.groupby("hospital_id", sort=False)
    }
    episodes_by_patient = {
        hospital_id: group.reset_index(drop=True)
        for hospital_id, group in _episode_windows().groupby("hospital_id", sort=False)
    }

    long_rows = []
    summary_rows = []
    feature_parts = []
    offsets = [0]
    for row in enriched.itertuples(index=False):
        hospital_id = str(row.hospital_id)
        current_time = float(row.current_lab_time_unix)
        episodes = episodes_by_patient.get(hospital_id)
        if episodes is None:
            raise ValueError(f"No admission episode for {target}/{row.video_id}")
        containing = episodes.loc[
            episodes["admission_time_unix"].le(current_time)
            & episodes["discharge_time_unix"].ge(current_time)
        ]
        if len(containing) != 1:
            raise ValueError(
                f"Expected one episode for {target}/{row.video_id}, found {len(containing)}"
            )
        episode = containing.iloc[0]
        patient_labs = labs_by_patient.get(hospital_id)
        if patient_labs is None:
            history = pd.DataFrame(columns=labs.columns)
        else:
            history = patient_labs.loc[
                patient_labs["timestamp_unix"].ge(episode["admission_time_unix"])
                & patient_labs["timestamp_unix"].le(episode["discharge_time_unix"])
                & patient_labs["timestamp_unix"].lt(current_time)
            ].copy()
        history = history.sort_values(["timestamp_unix", "value"]).reset_index(drop=True)
        values = history["value"].to_numpy(np.float64)
        delta_hours = (
            history["timestamp_unix"].to_numpy(np.float64) - current_time
        ) / 3600.0
        if len(delta_hours) and not np.all(delta_hours < 0.0):
            raise AssertionError(f"Non-prior history survived for {target}/{row.video_id}")
        value_features = scaler.transform(values)
        time_features = -np.log1p((-delta_hours) / HISTORY_TIME_SCALE_HOURS)
        features = np.column_stack((value_features, time_features)).astype(np.float32)
        feature_parts.append(features)
        offsets.append(offsets[-1] + len(features))

        for history_index, history_row in enumerate(history.itertuples(index=False)):
            long_rows.append(
                {
                    "target": target,
                    "hospital_id": hospital_id,
                    "video_id": str(row.video_id),
                    "split": str(row.split),
                    "current_lab_time_unix": current_time,
                    "episode_admission_time_unix": float(episode["admission_time_unix"]),
                    "episode_discharge_time_unix": float(episode["discharge_time_unix"]),
                    "history_index_oldest_first": history_index,
                    "history_lab_time_unix": float(history_row.timestamp_unix),
                    "history_value": float(history_row.value),
                    "history_item_name": str(history_row.item_name),
                    "history_unit": str(history_row.unit),
                    "history_minus_current_hours": float(delta_hours[history_index]),
                    "history_robust_scaled_raw_value": float(
                        value_features[history_index]
                    ),
                    "model_value_feature": float(value_features[history_index]),
                    "model_time_feature": float(time_features[history_index]),
                }
            )
        summary_rows.append(
            {
                "target": target,
                "hospital_id": hospital_id,
                "video_id": str(row.video_id),
                "split": str(row.split),
                "current_lab_time_unix": current_time,
                "history_count": len(history),
                "has_history": bool(len(history)),
                "oldest_history_delta_hours": (
                    float(delta_hours.min()) if len(delta_hours) else np.nan
                ),
                "nearest_history_delta_hours": (
                    float(delta_hours.max()) if len(delta_hours) else np.nan
                ),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    long_frame = pd.DataFrame(long_rows)
    summary_frame = pd.DataFrame(summary_rows)
    long_frame.to_csv(os.path.join(output_dir, f"{target}.csv"), index=False)
    summary_frame.to_csv(
        os.path.join(output_dir, f"{target}_summary.csv"), index=False
    )
    all_features = (
        np.concatenate(feature_parts, axis=0)
        if feature_parts
        else np.empty((0, 2), dtype=np.float32)
    )
    max_length = int(summary_frame["history_count"].max())
    np.savez_compressed(
        os.path.join(output_dir, f"{target}.npz"),
        video_ids=np.asarray(enriched["video_id"].astype(str).tolist(), dtype=str),
        offsets=np.asarray(offsets, dtype=np.int64),
        features=all_features,
        max_length=np.asarray([max_length], dtype=np.int64),
    )
    return summary_frame


def write_history_manifest(summaries, output_dir, scalers):
    combined = pd.concat(summaries, ignore_index=True)
    rows = []
    for (target, split), group in combined.groupby(["target", "split"], sort=True):
        counts = group["history_count"].to_numpy(np.int64)
        rows.append(
            {
                "target": target,
                "split": split,
                "videos": len(group),
                "videos_with_history": int((counts > 0).sum()),
                "history_coverage": float((counts > 0).mean()),
                "history_measurements": int(counts.sum()),
                "median_history_count": float(np.median(counts)),
                "max_history_count": int(counts.max()),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "history_coverage.csv"), index=False
    )
    with open(
        os.path.join(output_dir, "history_policy.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 1,
                "policy": HISTORY_POLICY,
                "analyte_scope": "same analyte as the prediction target",
                "analyte_source_policies": {
                    "po2": {
                        "canonical_item_name": PO2_CANONICAL_ITEM_NAME,
                        "excluded_item_names": list(PO2_EXCLUDED_ITEM_NAMES),
                        "enforcement": "exact item_name filter before sequence construction",
                    },
                    "oxyhemoglobin_fraction": {
                        "canonical_item_name": "氧合血红蛋白分数",
                        "canonical_unit": "%",
                        "excluded_specimen_pattern": "静脉",
                        "physical_range": [0.0, 100.0],
                        "enforcement": (
                            "exact item name, percent unit, non-venous specimen, "
                            "finite 0-100 value for labels and histories"
                        ),
                    },
                },
                "episode_scope": (
                    "same normalized hospital ID and unique closed admission-discharge interval "
                    "containing the current label timestamp"
                ),
                "temporal_rule": "history_lab_time_unix < current_lab_time_unix",
                "value_feature": (
                    "(raw history value - train-label median) / train-label IQR"
                ),
                "value_scalers": {
                    target: scaler.to_dict() for target, scaler in scalers.items()
                },
                "time_feature": (
                    f"-log1p((current_time-history_time)/{HISTORY_TIME_SCALE_HOURS} hours)"
                ),
                "sequence_policy": "all qualifying rows retained; no truncation",
                "pooling": "masked mean after per-measurement MLP",
                "missing_history": "zero pooled vector",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
