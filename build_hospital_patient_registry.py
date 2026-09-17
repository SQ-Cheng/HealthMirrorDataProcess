"""Build an audited hospital-patient registry from raw video directories.

The legacy ``PatientInfo`` extractor establishes the patient_info.txt layout,
while Exp2 supplies the shared placeholder-ID policy.  This script adds the
session-level validation and patient-level aggregation needed for a reliable
cross-mirror registry.
"""

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from study.exp2_lab_multimodal.config import PLACEHOLDER_HOSPITAL_IDS


DEFAULT_DATA_ROOT = Path("/root/shared/HealthMirrorDataset")
DEFAULT_LAB_CSV = Path(__file__).resolve().parent / "merged_lab_tests.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "patient_registry"
VALID_CANONICAL_ID_LENGTHS = {7, 8}
VIDEO_ONLY_REASONS = {"missing_video", "empty_video"}


def _canonical_id(value):
    """Normalize an ID to unpadded digits, rejecting known placeholders."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or not text.isdigit():
        return ""
    canonical = text.lstrip("0")
    if not canonical:
        return ""
    if (
        text in PLACEHOLDER_HOSPITAL_IDS
        or canonical in PLACEHOLDER_HOSPITAL_IDS
        or len(set(canonical)) == 1
    ):
        return ""
    return canonical


def _patient_info_fields(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    declared = re.search(r"^Patient ID:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    timestamp = re.search(
        r"^Session Timestamp:\s*(.*?)\s*$", text, flags=re.MULTILINE
    )
    payload = re.search(r"^Patient Info:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    if payload is None:
        raise ValueError("missing_patient_info_json")
    try:
        parsed = json.loads(payload.group(1))
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid_patient_info_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("invalid_patient_info_json_type")
    return {
        "declared_local_id": declared.group(1).strip() if declared else "",
        "session_timestamp_raw": timestamp.group(1).strip() if timestamp else "",
        "raw_hospital_patient_id": str(parsed.get("patient_id", "")).strip(),
    }


def _scan_sessions(data_root):
    rows = []
    directories = sorted(data_root.glob("mirror*_data/patient_*"))
    for patient_dir in directories:
        mirror_match = re.fullmatch(r"mirror(\d+)_data", patient_dir.parent.name)
        local_match = re.fullmatch(r"patient_(\d+)", patient_dir.name)
        mirror = f"mirror{int(mirror_match.group(1))}" if mirror_match else ""
        directory_local_id = local_match.group(1) if local_match else ""
        info_path = patient_dir / "patient_info.txt"
        video_path = patient_dir / "video.avi"
        reasons = []
        fields = {
            "declared_local_id": "", "session_timestamp_raw": "",
            "raw_hospital_patient_id": "",
        }
        if not mirror_match or not local_match:
            reasons.append("invalid_patient_directory_name")
        if not info_path.is_file():
            reasons.append("missing_patient_info")
        else:
            try:
                fields = _patient_info_fields(info_path)
            except (OSError, ValueError) as error:
                reasons.append(str(error))
        if fields["declared_local_id"]:
            try:
                same_local_id = int(fields["declared_local_id"]) == int(directory_local_id)
            except ValueError:
                same_local_id = False
            if not same_local_id:
                reasons.append("directory_and_declared_local_id_disagree")
        else:
            reasons.append("missing_declared_local_id")
        raw_hospital_id = fields["raw_hospital_patient_id"]
        hospital_id = _canonical_id(raw_hospital_id)
        if not raw_hospital_id:
            reasons.append("missing_hospital_patient_id")
        elif not raw_hospital_id.isdigit():
            reasons.append("nonnumeric_hospital_patient_id")
        elif not hospital_id:
            reasons.append("placeholder_hospital_patient_id")
        elif len(hospital_id) not in VALID_CANONICAL_ID_LENGTHS:
            reasons.append("unexpected_hospital_patient_id_length")
        if not video_path.is_file():
            video_bytes = 0
            video_status = "missing"
            reasons.append("missing_video")
        else:
            video_bytes = video_path.stat().st_size
            if video_bytes <= 0:
                video_status = "empty"
                reasons.append("empty_video")
            else:
                video_status = "nonempty"
        timestamp = pd.to_datetime(fields["session_timestamp_raw"], errors="coerce")
        patient_id_reasons = [reason for reason in reasons if reason not in VIDEO_ONLY_REASONS]
        rows.append({
            "mirror": mirror,
            "lab_patient_id": str(int(directory_local_id)).zfill(6) if directory_local_id else "",
            "hospital_patient_id": hospital_id,
            "raw_hospital_patient_id": raw_hospital_id,
            "declared_local_id": fields["declared_local_id"],
            "session_timestamp": timestamp,
            "session_timestamp_valid": int(pd.notna(timestamp)),
            "video_bytes": video_bytes,
            "video_status": video_status,
            "patient_directory": str(patient_dir),
            "patient_info_path": str(info_path),
            "video_path": str(video_path),
            "filter_reasons": ";".join(dict.fromkeys(reasons)),
            "is_valid_patient_id": int(not patient_id_reasons),
            "is_valid_video_session": int(not reasons),
        })
    return pd.DataFrame(rows)


def _load_lab_index(path):
    columns = ["首页病案号", "检验项名称", "报告时间"]
    labs = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    # Three source IDs are wrapped in asterisks; remove only such wrappers.
    raw = labs["首页病案号"].str.strip().str.replace(
        r"^\*([0-9]+)\*$", r"\1", regex=True
    )
    labs["hospital_patient_id"] = raw.map(_canonical_id)
    labs["report_time"] = pd.to_datetime(labs["报告时间"], errors="coerce")
    valid = labs[labs.hospital_patient_id.ne("")].copy()
    summary = valid.groupby("hospital_patient_id", as_index=False).agg(
        lab_row_count=("检验项名称", "size"),
        lab_item_count=("检验项名称", "nunique"),
        first_lab_report_time=("report_time", "min"),
        last_lab_report_time=("report_time", "max"),
    )
    return labs, summary


def _join_unique(series):
    return ";".join(sorted(set(str(value) for value in series if str(value))))


def _build_registry(patient_sessions, lab_summary):
    registry = patient_sessions.groupby("hospital_patient_id", as_index=False).agg(
        patient_info_session_count=("patient_directory", "size"),
        video_session_count=("is_valid_video_session", "sum"),
        missing_video_session_count=(
            "video_status", lambda values: int(values.eq("missing").sum()),
        ),
        empty_video_session_count=(
            "video_status", lambda values: int(values.eq("empty").sum()),
        ),
        mirror_count=("mirror", "nunique"),
        mirrors=("mirror", _join_unique),
        mirror_local_ids=(
            "mirror_local_id", _join_unique,
        ),
        valid_session_timestamp_count=("session_timestamp_valid", "sum"),
        first_session_timestamp=("session_timestamp", "min"),
        last_session_timestamp=("session_timestamp", "max"),
        total_video_bytes=("video_bytes", "sum"),
    )
    registry = registry.merge(lab_summary, on="hospital_patient_id", how="left")
    registry["has_lab_data"] = registry.lab_row_count.notna().astype(int)
    for column in ("lab_row_count", "lab_item_count"):
        registry[column] = registry[column].fillna(0).astype(int)
    registry["hospital_patient_id_numeric"] = pd.to_numeric(
        registry.hospital_patient_id, errors="raise"
    )
    registry = registry.sort_values("hospital_patient_id_numeric").drop(
        columns="hospital_patient_id_numeric"
    ).reset_index(drop=True)
    registry["hospital_patient_id"] = registry.hospital_patient_id.str.zfill(10)
    columns = [
        "hospital_patient_id", "has_lab_data", "patient_info_session_count",
        "video_session_count", "missing_video_session_count",
        "empty_video_session_count",
        "mirror_count", "mirrors", "mirror_local_ids",
        "valid_session_timestamp_count", "first_session_timestamp",
        "last_session_timestamp", "total_video_bytes", "lab_row_count",
        "lab_item_count", "first_lab_report_time", "last_lab_report_time",
    ]
    return registry[columns]


def build_registry(data_root, lab_csv, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = _scan_sessions(data_root)
    patient_sessions = sessions[sessions.is_valid_patient_id.eq(1)].copy()
    patient_sessions["mirror_local_id"] = (
        patient_sessions.mirror + ":" + patient_sessions.lab_patient_id
    )
    valid = sessions[sessions.is_valid_video_session.eq(1)].copy()
    excluded = sessions[sessions.is_valid_video_session.eq(0)].copy()
    labs, lab_summary = _load_lab_index(lab_csv)
    registry = _build_registry(patient_sessions, lab_summary)
    without_labs = registry[registry.has_lab_data.eq(0)].copy()

    valid.drop(
        columns=["filter_reasons", "is_valid_patient_id", "is_valid_video_session"]
    ).to_csv(
        output_dir / "hospital_patient_video_sessions.csv", index=False
    )
    excluded.drop(columns="is_valid_video_session").to_csv(
        output_dir / "excluded_patient_video_sessions.csv", index=False
    )
    registry.to_csv(output_dir / "hospital_patient_ids.csv", index=False)
    without_labs.to_csv(
        output_dir / "hospital_patient_ids_without_lab_data.csv", index=False
    )

    summary_rows = [
        ("scanned_patient_directories", len(sessions)),
        ("patient_id_eligible_sessions", len(patient_sessions)),
        ("retained_video_sessions", len(valid)),
        ("excluded_video_sessions", len(excluded)),
        ("retained_hospital_patient_ids", len(registry)),
        ("hospital_patient_ids_with_lab_data", int(registry.has_lab_data.sum())),
        ("hospital_patient_ids_without_lab_data", len(without_labs)),
        ("lab_rows", len(labs)),
        ("lab_hospital_patient_ids", labs.hospital_patient_id.replace("", pd.NA).nunique()),
    ]
    reason_counts = (
        excluded.filter_reasons.str.split(";").explode().value_counts().sort_index()
    )
    summary_rows.extend((f"excluded_reason:{reason}", int(count)) for reason, count in reason_counts.items())
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
        output_dir / "registry_summary.csv", index=False
    )
    print(
        f"[complete] sessions={len(sessions)} retained={len(valid)} "
        f"patient_id_eligible={len(patient_sessions)} excluded={len(excluded)} "
        f"hospital_ids={len(registry)} "
        f"with_labs={int(registry.has_lab_data.sum())} "
        f"without_labs={len(without_labs)} output={output_dir}",
        flush=True,
    )
    return registry, without_labs, valid, excluded


def main():
    parser = argparse.ArgumentParser(
        description="Build an audited hospital patient registry from raw videos"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--lab-csv", type=Path, default=DEFAULT_LAB_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_registry(args.data_root, args.lab_csv, args.output_dir)


if __name__ == "__main__":
    main()
