"""Build one nearest 24-hour lab label per current raw video and target."""

from collections import Counter, defaultdict
import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from study.exp2_lab_multimodal.build_dataset import (
    LAB_REPORT_TIMEZONE,
    _extract_numeric,
    _normalize_hospital_id,
    _parse_datetime_to_unix,
    _read_merged_patient_info,
)
from study.exp2_lab_multimodal.config import DATA_ROOT, LAB_CSV

from .config import (
    LAB_MATCH_MAX_DELTA_HOURS,
    SCORE_DEFINITIONS,
)


TARGET_ANALYTES = {
    "oxyhemoglobin_fraction": "oxyhemoglobin_fraction",
    "lactate_high": "lactate",
    "urea_high": "urea",
    "troponin_high": "troponin",
    "platelet_count_low": "platelet_count",
    "hemoglobin_low": "hemoglobin",
    "aa_po2_ratio_low": "aa_po2_ratio",
    "creatinine_high": "creatinine",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video_time_bounds(path):
    timestamps = []
    invalid_rows = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            try:
                timestamp = float(line.rsplit(",", 1)[-1].strip())
            except (IndexError, ValueError):
                invalid_rows += 1
                continue
            if np.isfinite(timestamp):
                timestamps.append(timestamp)
            else:
                invalid_rows += 1
    if not timestamps:
        return None
    values = np.asarray(timestamps, dtype=np.float64)
    return {
        "capture_start_unix": float(values.min()),
        "capture_end_unix": float(values.max()),
        "timestamp_rows": int(len(values)),
        "invalid_timestamp_rows": int(invalid_rows),
        "nonmonotonic_steps": int(np.count_nonzero(np.diff(values) < 0)),
    }


LAB_SOURCE_DEFINITIONS = {
    "oxyhemoglobin_fraction": {
        "items": ("氧合血红蛋白分数", "氧合血红蛋白"),
        "canonical_unit": "%",
        "valid_range": (0.0, 100.0),
        "exclude_venous": True,
        "unit_conversion": {"%": 1.0},
    },
    "lactate": {
        "items": ("*乳酸浓度", "乳酸浓度", "乳酸"),
        "canonical_unit": "mmol/L",
        "valid_range": (0.05, 30.0),
        "exclude_venous": False,
        "unit_conversion": {"mmol/L": 1.0},
    },
    "urea": {
        "items": ("*尿素(Urea)测定",),
        "canonical_unit": "mmol/L",
        "valid_range": (0.1, 100.0),
        "exclude_venous": False,
        "unit_conversion": {"mmol/L": 1.0},
    },
    "troponin": {
        "items": (
            "*肌钙蛋白Ⅰ(hsTnI)测定",
            "肌钙蛋白Ⅰ(hsTnI)测定",
            "*全血肌钙蛋白Ⅰ",
            "全血肌钙蛋白Ⅰ",
            "*高敏肌钙蛋白Ⅰ测定",
            "肌钙蛋白Ⅰ(TnI)测定",
        ),
        "canonical_unit": "ng/L",
        "valid_range": (0.0, 500000.0),
        "exclude_venous": False,
        "unit_conversion": {
            "ng/L": 1.0,
            "pg/mL": 1.0,
            "ug/L": 1000.0,
            "ng/mL": 1000.0,
        },
    },
    "platelet_count": {
        "items": ("*血小板", "血小板"),
        "canonical_unit": "10^9/L",
        "valid_range": (1.0, 2000.0),
        "exclude_venous": False,
        "unit_conversion": {"*10^9/L": 1.0, "10^9/L": 1.0, "G/L": 1.0},
    },
    "hemoglobin": {
        "items": ("*血红蛋白", "血红蛋白", "总血红蛋白"),
        "canonical_unit": "g/L",
        "valid_range": (20.0, 250.0),
        "exclude_venous": False,
        "unit_conversion": {"g/L": 1.0, "g/dL": 10.0},
    },
    "aa_po2_ratio": {
        "items": (
            "动脉氧分压与肺泡氧分压之比",
            "动脉血氧分压与肺泡内氧分压之比",
        ),
        "canonical_unit": "%",
        "valid_range": (1.0, 800.0),
        "exclude_venous": True,
        "unit_conversion": {
            "动脉氧分压与肺泡氧分压之比 [%]": 1.0,
            "动脉血氧分压与肺泡内氧分压之比 [unitless]": 100.0,
        },
    },
    "creatinine": {
        "items": ("*肌酐(Cr)测定", "*肌酐(Cr)测定-苦味酸法", "*肌酐"),
        "canonical_unit": "umol/L",
        "valid_range": (5.0, 2000.0),
        "exclude_venous": False,
        "unit_conversion": {"umol/L": 1.0},
    },
    "total_bilirubin": {
        "items": (
            "*总胆红素(T-Bil)测定", "总胆红素(T-Bil)测定", "总胆红素",
        ),
        "canonical_unit": "umol/L",
        "valid_range": (0.1, 1000.0),
        "exclude_venous": False,
        "unit_conversion": {"umol/L": 1.0},
    },
}


def _canonical_values(analyte, items, values, units):
    values = pd.to_numeric(values, errors="coerce")
    units = (
        units.astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.lower()
        .str.replace("μ", "u", regex=False)
        .str.replace("µ", "u", regex=False)
        .str.replace("∧", "^", regex=False)
    )
    if analyte == "hemoglobin":
        supported = units.isin(("g/l", "g/dl"))
        is_gdl = units.eq("g/dl")
        return values.where(~is_gdl, values * 10.0), supported
    if analyte == "troponin":
        factors = units.map({
            "ng/l": 1.0,
            "pg/ml": 1.0,
            "ug/l": 1000.0,
            "ng/ml": 1000.0,
        })
        return values * factors, factors.notna()
    if analyte == "platelet_count":
        supported = units.isin(("*10^9/l", "10^9/l", "g/l"))
        return values, supported
    if analyte == "aa_po2_ratio":
        fraction_item = items.eq("动脉血氧分压与肺泡内氧分压之比")
        percent_item = items.eq("动脉氧分压与肺泡氧分压之比")
        supported = (fraction_item & units.eq("")) | (percent_item & units.eq("%"))
        return values.where(~fraction_item, values * 100.0), supported
    expected = {
        "oxyhemoglobin_fraction": "%",
        "lactate": "mmol/l",
        "urea": "mmol/l",
        "creatinine": "umol/l",
        "total_bilirubin": "umol/l",
    }[analyte]
    return values, units.eq(expected)


def _load_lab_data(targets, output_dir):
    requested_analytes = {TARGET_ANALYTES[target] for target in targets}
    columns = [
        "首页病案号", "首页性别", "检验套名称", "检验项名称",
        "检验值(文本)", "单位", "标本名称", "报告时间",
    ]
    raw = pd.read_csv(
        LAB_CSV, dtype=str, keep_default_na=False, usecols=columns,
    )
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["timestamp_unix"] = _parse_datetime_to_unix(raw["报告时间"])
    raw["numeric_value"] = _extract_numeric(
        raw["检验值(文本)"].str.replace(",", "", regex=False)
    )
    raw["censored"] = raw["检验值(文本)"].str.match(
        r"^\s*[<>≤≥＜＞]", na=False,
    )

    lab_frames, policies = [], {}
    for analyte in sorted(requested_analytes):
        definition = LAB_SOURCE_DEFINITIONS[analyte]
        selected = raw.loc[raw["检验项名称"].isin(definition["items"])].copy()
        values, unit_supported = _canonical_values(
            analyte,
            selected["检验项名称"],
            selected["numeric_value"],
            selected["单位"],
        )
        selected["value"] = values
        selected["unit_supported"] = unit_supported
        selected["non_venous"] = ~selected["标本名称"].str.contains(
            "静脉", regex=False,
        )
        selected["in_valid_range"] = selected["value"].between(
            *definition["valid_range"], inclusive="both",
        )
        selected["valid"] = (
            selected["hospital_id"].ne("")
            & selected["timestamp_unix"].notna()
            & selected["value"].notna()
            & ~selected["censored"]
            & selected["unit_supported"]
            & selected["in_valid_range"]
            & (
                selected["non_venous"]
                if definition["exclude_venous"]
                else True
            )
        )
        retained = selected.loc[selected["valid"]].copy()
        conflicts = (
            retained.groupby(["hospital_id", "timestamp_unix"])["value"]
            .nunique().gt(1)
        )
        retained = retained.groupby(
            ["hospital_id", "timestamp_unix"], as_index=False,
        ).agg(
            value=("value", "median"),
            item_name=("检验项名称", lambda values: "|".join(sorted(set(values)))),
        )
        retained["analyte"] = analyte
        retained["unit"] = definition["canonical_unit"]
        lab_frames.append(retained[
            ["hospital_id", "analyte", "value", "timestamp_unix", "unit", "item_name"]
        ])
        policies[analyte] = {
            "accepted_item_names": list(definition["items"]),
            "canonical_item_name": definition["items"][0],
            "canonical_unit": definition["canonical_unit"],
            "excluded_item_names": [],
            "exclude_venous_specimens": definition["exclude_venous"],
            "valid_range": list(definition["valid_range"]),
            "unit_conversion": definition["unit_conversion"],
            "source_rows": int(len(selected)),
            "retained_source_rows": int(selected["valid"].sum()),
            "retained_patient_timestamp_events": int(len(retained)),
            "retained_patients": int(retained["hospital_id"].nunique()),
            "source_rows_by_item": {
                str(name): int(count)
                for name, count in selected["检验项名称"].value_counts().items()
            },
            "retained_source_rows_by_item": {
                str(name): int(count)
                for name, count in selected.loc[
                    selected["valid"], "检验项名称"
                ].value_counts().items()
            },
            "source_units": {
                str(name): int(count)
                for name, count in selected["单位"].value_counts().items()
            },
            "conflicting_patient_timestamps_collapsed_by_median": int(conflicts.sum()),
            "enforcement": (
                "explicit item aliases, canonical unit conversion, specimen policy, "
                "finite physical range, censored-value exclusion, deterministic median "
                "for duplicate patient timestamps"
            ),
        }
    labs = pd.concat(lab_frames, ignore_index=True)
    labs = labs.sort_values(
        ["hospital_id", "analyte", "timestamp_unix", "value"],
    ).reset_index(drop=True)
    labs.to_csv(os.path.join(output_dir, "lab_timeseries.csv"), index=False)

    sex_source = raw[["hospital_id", "首页性别"]].copy()
    sex_lookup = {}
    for hospital_id, group in sex_source[
        sex_source["hospital_id"].ne("")
    ].groupby("hospital_id", sort=False):
        values = group["首页性别"].astype(str).str.strip()
        values = values[~values.isin(("", "nan", "None"))]
        sex_lookup[str(hospital_id)] = values.iloc[0] if len(values) else ""
    quality = {
        "lab_report_time": {
            "source_column": "报告时间",
            "source_timezone": LAB_REPORT_TIMEZONE,
            "stored_representation": "UTC Unix seconds",
        },
        "analyte_source_policies": policies,
    }
    return labs, sex_lookup, quality


def _interval_delta_seconds(timestamp, start, end):
    if timestamp < start:
        return start - timestamp
    if timestamp > end:
        return timestamp - end
    return 0.0


def _nearest_measurement(measurements, start, end):
    midpoint = (start + end) / 2.0
    candidates = []
    for source_order, (timestamp, value) in enumerate(measurements):
        delta = _interval_delta_seconds(timestamp, start, end)
        if delta <= LAB_MATCH_MAX_DELTA_HOURS * 3600.0:
            candidates.append(
                (delta, abs(timestamp - midpoint), timestamp, source_order, value)
            )
    if not candidates:
        return None
    delta, _, timestamp, _, value = min(candidates, key=lambda row: row[:4])
    if timestamp < start:
        signed_delta = -delta
    elif timestamp > end:
        signed_delta = delta
    else:
        signed_delta = 0.0
    return {
        "value": float(value),
        "timestamp_unix": float(timestamp),
        "delta_h": float(delta / 3600.0),
        "signed_delta_h": float(signed_delta / 3600.0),
    }


def _binary_label(target, value, sex):
    definition = SCORE_DEFINITIONS[target]
    threshold = (
        definition["threshold"]["male"]
        if target == "hemoglobin_low" and sex == "男"
        else (
            definition["threshold"]["other"]
            if target == "hemoglobin_low"
            else definition["threshold"]
        )
    )
    if definition["direction"] == "low":
        return int(value < threshold)
    if definition["direction"] == "high":
        return int(value > threshold)
    raise ValueError(f"Unsupported direction for {target}")


def validate_analyte_source_policies(quality, targets):
    """Verify that every requested target used its complete canonical source policy."""
    policies = quality.get("analyte_source_policies", {})
    for target in targets:
        analyte = TARGET_ANALYTES[target]
        expected = LAB_SOURCE_DEFINITIONS[analyte]
        actual = policies.get(analyte, {})
        problems = []
        if actual.get("accepted_item_names") != list(expected["items"]):
            problems.append("accepted_item_names")
        if actual.get("canonical_unit") != expected["canonical_unit"]:
            problems.append("canonical_unit")
        if actual.get("exclude_venous_specimens") is not expected["exclude_venous"]:
            problems.append("exclude_venous_specimens")
        if actual.get("valid_range") != list(expected["valid_range"]):
            problems.append("valid_range")
        if actual.get("unit_conversion") != expected["unit_conversion"]:
            problems.append("unit_conversion")
        if int(actual.get("retained_patient_timestamp_events", 0)) <= 0:
            problems.append("retained_patient_timestamp_events")
        if problems:
            raise RuntimeError(
                f"Invalid source policy for {target}/{analyte}: "
                f"fields={problems}, policy={actual}"
            )


def build_raw_video_source(output_dir, targets):
    """Write the current all-video pool and one nearest value per target."""
    os.makedirs(output_dir, exist_ok=True)
    unknown = sorted(set(targets) - set(TARGET_ANALYTES))
    if unknown:
        raise ValueError(f"Unsupported raw-video targets: {unknown}")

    labs, sex_lookup, upstream_quality = _load_lab_data(targets, output_dir)
    info_lookup = _read_merged_patient_info()
    measurements = defaultdict(lambda: defaultdict(list))
    for row in labs.itertuples(index=False):
        measurements[str(row.hospital_id)][str(row.analyte)].append(
            (float(row.timestamp_unix), float(row.value))
        )
    all_event_times = {
        hospital_id: np.asarray(
            sorted(
                {
                    timestamp
                    for analyte_values in analytes.values()
                    for timestamp, _ in analyte_values
                }
            ),
            dtype=np.float64,
        )
        for hospital_id, analytes in measurements.items()
    }

    raw_paths = sorted(
        glob.glob(
            os.path.join(DATA_ROOT, "mirror*_data", "patient_*", "video.avi")
        )
    )
    audit_rows = []
    base_rows = []
    video_rows = []
    skip_counts = Counter()
    for video_path in raw_paths:
        match = re.search(
            r"/(mirror\d+)_data/patient_(\d+)/video\.avi$", video_path
        )
        if match is None:
            skip_counts["path_parse_failed"] += 1
            audit_rows.append(
                {"video_path": video_path, "status": "path_parse_failed"}
            )
            continue
        mirror = match.group(1)
        lab_patient_id = int(match.group(2))
        video_id = f"{mirror}_patient_{lab_patient_id:06d}"
        info = info_lookup.get((mirror, lab_patient_id))
        if info is None:
            skip_counts["missing_patient_mapping"] += 1
            audit_rows.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "status": "missing_patient_mapping",
                }
            )
            continue
        hospital_id = _normalize_hospital_id(
            info.get("Hospital_Patient_ID", "")
        )
        if hospital_id == "":
            skip_counts["invalid_or_placeholder_hospital_id"] += 1
            audit_rows.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "status": "invalid_or_placeholder_hospital_id",
                }
            )
            continue
        timestamp_path = video_path + ".ts"
        if not os.path.isfile(timestamp_path):
            skip_counts["missing_video_timestamp_file"] += 1
            audit_rows.append(
                {
                    "video_id": video_id,
                    "hospital_id": hospital_id,
                    "video_path": video_path,
                    "status": "missing_video_timestamp_file",
                }
            )
            continue
        bounds = _read_video_time_bounds(timestamp_path)
        if bounds is None:
            skip_counts["invalid_video_timestamps"] += 1
            audit_rows.append(
                {
                    "video_id": video_id,
                    "hospital_id": hospital_id,
                    "video_path": video_path,
                    "status": "invalid_video_timestamps",
                }
            )
            continue
        start = bounds["capture_start_unix"]
        end = bounds["capture_end_unix"]
        patient_events = all_event_times.get(hospital_id)
        if patient_events is None or not len(patient_events):
            skip_counts["patient_without_supported_lab"] += 1
            audit_rows.append(
                {
                    "video_id": video_id,
                    "hospital_id": hospital_id,
                    "video_path": video_path,
                    "status": "patient_without_supported_lab",
                    **bounds,
                }
            )
            continue
        deltas = np.where(
            patient_events < start,
            start - patient_events,
            np.where(patient_events > end, patient_events - end, 0.0),
        )
        nearest_any_delta_h = float(deltas.min() / 3600.0)
        if nearest_any_delta_h > LAB_MATCH_MAX_DELTA_HOURS:
            skip_counts["supported_lab_outside_24h"] += 1
            audit_rows.append(
                {
                    "video_id": video_id,
                    "hospital_id": hospital_id,
                    "video_path": video_path,
                    "status": "supported_lab_outside_24h",
                    "nearest_any_lab_delta_h": nearest_any_delta_h,
                    **bounds,
                }
            )
            continue

        sex = sex_lookup.get(hospital_id, "")
        row = {
            "sample_id": f"raw_video_{video_id}",
            "event_type": "raw_video_nearest_lab_per_target",
            "hospital_id": hospital_id,
            "video_id": video_id,
            "mirror": mirror,
            "lab_patient_id": lab_patient_id,
            "capture_time_unix": (start + end) / 2.0,
            "capture_start_unix": start,
            "capture_end_unix": end,
            "sex": sex,
        }
        available_targets = []
        for target in targets:
            analyte = TARGET_ANALYTES[target]
            nearest = _nearest_measurement(
                measurements[hospital_id].get(analyte, ()), start, end
            )
            prefix = analyte
            row[target] = np.nan
            row[f"{prefix}_value"] = np.nan
            row[f"{prefix}_delta_h"] = np.nan
            row[f"{prefix}_signed_delta_h"] = np.nan
            row[f"{prefix}_lab_time_unix"] = np.nan
            if nearest is None:
                continue
            row[target] = _binary_label(target, nearest["value"], sex)
            row[f"{prefix}_value"] = nearest["value"]
            row[f"{prefix}_delta_h"] = nearest["delta_h"]
            row[f"{prefix}_signed_delta_h"] = nearest["signed_delta_h"]
            row[f"{prefix}_lab_time_unix"] = nearest["timestamp_unix"]
            available_targets.append(target)
        low_bp = pd.to_numeric(
            info.get("Low_Blood_Pressure", np.nan), errors="coerce"
        )
        high_bp = pd.to_numeric(
            info.get("High_Blood_Pressure", np.nan), errors="coerce"
        )
        base_rows.append(row)
        video_rows.append(
            {
                "video_id": video_id,
                "hospital_id": hospital_id,
                "mirror": mirror,
                "lab_patient_id": lab_patient_id,
                "video_path": video_path,
                "capture_time_unix": row["capture_time_unix"],
                "capture_start_unix": start,
                "capture_end_unix": end,
                "timestamp_rows": bounds["timestamp_rows"],
                "invalid_timestamp_rows": bounds["invalid_timestamp_rows"],
                "nonmonotonic_steps": bounds["nonmonotonic_steps"],
                "low_blood_pressure": low_bp,
                "high_blood_pressure": high_bp,
                "blood_pressure_valid": 0,
            }
        )
        audit_rows.append(
            {
                "video_id": video_id,
                "hospital_id": hospital_id,
                "video_path": video_path,
                "status": "retained_24h_pool",
                "nearest_any_lab_delta_h": nearest_any_delta_h,
                "available_targets": ",".join(available_targets),
                **bounds,
            }
        )

    base_manifest = pd.DataFrame(base_rows).sort_values("video_id").reset_index(
        drop=True
    )
    video_summary = pd.DataFrame(video_rows).sort_values("video_id").reset_index(
        drop=True
    )
    audit = pd.DataFrame(audit_rows)
    if base_manifest["video_id"].duplicated().any():
        raise AssertionError("Raw-video source contains duplicate video IDs")

    target_counts = {}
    for target in targets:
        labels = pd.to_numeric(base_manifest[target], errors="coerce")
        target_counts[target] = {
            "videos": int(labels.notna().sum()),
            "patients": int(
                base_manifest.loc[labels.notna(), "hospital_id"].nunique()
            ),
            "positive_videos": int(labels.eq(1).sum()),
            "negative_videos": int(labels.eq(0).sum()),
        }
    retained_count = int(len(base_manifest))
    counts = {
        "raw_video_files": int(len(raw_paths)),
        "retained_24h_pool_videos": retained_count,
        "retained_24h_pool_patients": int(
            base_manifest["hospital_id"].nunique()
        ),
        "skips": dict(skip_counts),
        "targets": target_counts,
    }
    report = {
        "schema_version": 2,
        "experiment": "exp2_raw_video_nearest_lab_source",
        "lab_report_time": upstream_quality["lab_report_time"],
        "video_match_policy": {
            "mode": "raw_video_interval_nearest_lab_per_target",
            "video_time_source": "video.avi.ts",
            "capture_interval": "[minimum valid frame timestamp, maximum valid frame timestamp]",
            "maximum_delta_hours": LAB_MATCH_MAX_DELTA_HOURS,
            "interval_distance": (
                "zero inside capture interval; otherwise distance to nearest boundary"
            ),
            "target_label_policy": (
                "one closest in-window measurement per video and target; "
                "interval distance, then midpoint distance, then timestamp"
            ),
            "session_csv_required": False,
        },
        "analyte_source_policies": upstream_quality.get(
            "analyte_source_policies", {}
        ),
        "counts": counts,
        "source_fingerprints": {
            "lab_timeseries_cache": {
                "path": os.path.join(output_dir, "lab_timeseries.csv"),
                "sha256": _sha256(os.path.join(output_dir, "lab_timeseries.csv")),
            },
            "merged_lab_tests": {
                "path": LAB_CSV,
                "size_bytes": os.path.getsize(LAB_CSV),
                "mtime_ns": os.stat(LAB_CSV).st_mtime_ns,
            },
        },
        "files": {
            "base_manifest": "base_manifest.csv",
            "video_summary": "video_summary.csv",
            "raw_video_audit": "raw_video_audit.csv",
        },
    }
    base_manifest.to_csv(
        os.path.join(output_dir, "base_manifest.csv"), index=False
    )
    video_summary.to_csv(
        os.path.join(output_dir, "video_summary.csv"), index=False
    )
    audit.to_csv(os.path.join(output_dir, "raw_video_audit.csv"), index=False)
    with open(
        os.path.join(output_dir, "data_quality_report.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        f"Built raw-video source: pool={retained_count} "
        f"patients={counts['retained_24h_pool_patients']} targets={target_counts}",
        flush=True,
    )
    return base_manifest, video_summary, report
