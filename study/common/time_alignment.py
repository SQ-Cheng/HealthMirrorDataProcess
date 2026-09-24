"""Canonical parsing and validation of video session times.

Clinical matching must use ``patient_info.txt`` Session Timestamp. Frame
timestamps are an independent recorder clock and are used only to estimate the
recording duration and to expose source-clock discrepancies for auditing.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_TIMEZONE = "Asia/Shanghai"
MAX_CONTIGUOUS_FRAME_GAP_SECONDS = 10.0
MAX_VIDEO_DURATION_SECONDS = 30.0 * 60.0
SOURCE_DELTA_WARNING_SECONDS = 5.0 * 60.0


class TimeAlignmentError(ValueError):
    """A machine-readable video-time validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text or not text.isdigit():
        return ""
    return text.lstrip("0") or "0"


def _extract_line(text: str, field: str) -> str:
    match = re.search(
        rf"^{re.escape(field)}:\s*(.*?)\s*$", text, flags=re.MULTILINE
    )
    return match.group(1).strip() if match else ""


def _parse_patient_info_payload(value: str) -> dict:
    if not value:
        return {}
    try:
        payload = json.loads(value)
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def local_naive_to_unix(value, timezone: str = DEFAULT_TIMEZONE) -> float:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise TimeAlignmentError("invalid_session_timestamp", f"Invalid local time: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        )
    else:
        timestamp = timestamp.tz_convert(timezone)
    return float(timestamp.tz_convert("UTC").timestamp())


def unix_to_local_naive(value: float, timezone: str = DEFAULT_TIMEZONE) -> pd.Timestamp:
    return pd.Timestamp(value, unit="s", tz="UTC").tz_convert(timezone).tz_localize(None)


def read_session_metadata(
    patient_info_path,
    *,
    expected_local_id=None,
    expected_hospital_id=None,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict:
    path = Path(patient_info_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TimeAlignmentError("missing_patient_info", str(exc)) from exc

    local_id = _canonical_id(_extract_line(text, "Patient ID"))
    expected_local = _canonical_id(expected_local_id)
    if not local_id:
        raise TimeAlignmentError("invalid_session_local_id", f"Missing Patient ID in {path}")
    if expected_local and local_id != expected_local:
        raise TimeAlignmentError(
            "session_local_id_mismatch",
            f"Patient ID {local_id} != expected {expected_local} in {path}",
        )

    session_text = _extract_line(text, "Session Timestamp")
    session_unix = local_naive_to_unix(session_text, timezone)
    session_local = unix_to_local_naive(session_unix, timezone)
    now_local = pd.Timestamp.now(tz=timezone).tz_localize(None)
    if session_local < pd.Timestamp("2018-01-01") or session_local > now_local + pd.Timedelta(days=366):
        raise TimeAlignmentError(
            "implausible_session_timestamp",
            f"Session Timestamp outside plausible range in {path}: {session_text}",
        )

    payload = _parse_patient_info_payload(_extract_line(text, "Patient Info"))
    embedded_hospital_id = _canonical_id(payload.get("patient_id", ""))
    expected_hospital = _canonical_id(expected_hospital_id)
    if not embedded_hospital_id:
        raise TimeAlignmentError(
            "invalid_session_hospital_id", f"Missing embedded patient_id in {path}"
        )
    if expected_hospital and embedded_hospital_id != expected_hospital:
        raise TimeAlignmentError(
            "session_hospital_id_mismatch",
            f"Embedded patient_id {embedded_hospital_id} != expected {expected_hospital} in {path}",
        )

    return {
        "session_time_text": session_text,
        "session_time_local": session_local,
        "session_time_unix": session_unix,
        "session_local_patient_id": local_id,
        "session_hospital_id": embedded_hospital_id,
        "session_validation": "patient_and_hospital_ids_match",
    }


def read_frame_timestamp_diagnostics(
    timestamp_path,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict:
    path = Path(timestamp_path)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise TimeAlignmentError("missing_video_timestamp_file", str(exc)) from exc

    now_unix = local_naive_to_unix(
        pd.Timestamp.now(tz=timezone).tz_localize(None) + pd.Timedelta(days=366),
        timezone,
    )
    earliest_unix = local_naive_to_unix("2018-01-01", timezone)
    parsed = []
    invalid_rows = 0
    implausible_rows = 0
    for line in lines[1:]:
        parts = line.rsplit(",", 1)
        try:
            frame_index = int(parts[0].strip())
            timestamp = float(parts[1].strip())
        except (IndexError, ValueError):
            invalid_rows += 1
            continue
        if not np.isfinite(timestamp):
            invalid_rows += 1
            continue
        if timestamp < earliest_unix or timestamp > now_unix:
            implausible_rows += 1
            continue
        parsed.append((frame_index, timestamp))
    if len(parsed) < 2:
        raise TimeAlignmentError(
            "invalid_video_timestamps", f"Fewer than two valid frame timestamps in {path}"
        )

    segments = []
    current = [parsed[0]]
    discontinuities = 0
    nonmonotonic = 0
    for item in parsed[1:]:
        previous = current[-1]
        frame_step = item[0] - previous[0]
        time_step = item[1] - previous[1]
        if frame_step <= 0 or time_step <= 0:
            nonmonotonic += 1
        if (
            frame_step > 0
            and time_step > 0
            and time_step <= MAX_CONTIGUOUS_FRAME_GAP_SECONDS
        ):
            current.append(item)
        else:
            discontinuities += 1
            segments.append(current)
            current = [item]
    segments.append(current)
    segment = max(segments, key=lambda values: (len(values), values[-1][0] - values[0][0]))
    if len(segment) < 2:
        raise TimeAlignmentError(
            "invalid_video_timestamp_sequence", f"No contiguous frame timestamp segment in {path}"
        )
    duration = float(segment[-1][1] - segment[0][1])
    if not 0.0 < duration <= MAX_VIDEO_DURATION_SECONDS:
        raise TimeAlignmentError(
            "implausible_video_duration",
            f"Video duration {duration:.3f}s outside (0,{MAX_VIDEO_DURATION_SECONDS}] in {path}",
        )
    return {
        "frame_timestamp_start_unix": float(segment[0][1]),
        "frame_timestamp_end_unix": float(segment[-1][1]),
        "video_duration_seconds": duration,
        "timestamp_rows": int(len(parsed)),
        "invalid_timestamp_rows": int(invalid_rows),
        "implausible_timestamp_rows": int(implausible_rows),
        "timestamp_discontinuities": int(discontinuities),
        "timestamp_rows_outside_primary_segment": int(len(parsed) - len(segment)),
        "nonmonotonic_steps": int(nonmonotonic),
    }


def read_video_session(
    video_path,
    *,
    expected_local_id=None,
    expected_hospital_id=None,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict:
    video_path = Path(video_path)
    session = read_session_metadata(
        video_path.with_name("patient_info.txt"),
        expected_local_id=expected_local_id,
        expected_hospital_id=expected_hospital_id,
        timezone=timezone,
    )
    frame = read_frame_timestamp_diagnostics(
        Path(str(video_path) + ".ts"), timezone=timezone
    )
    start = session["session_time_unix"]
    end = start + frame["video_duration_seconds"]
    signed_delta = frame["frame_timestamp_start_unix"] - start
    return {
        **session,
        **frame,
        "capture_start_unix": float(start),
        "capture_end_unix": float(end),
        "capture_midpoint_unix": float((start + end) / 2.0),
        "video_time_source": "patient_info.txt:Session Timestamp",
        "frame_timestamp_role": "duration_and_diagnostics_only",
        "frame_timestamp_source_delta_seconds": float(signed_delta),
        "frame_timestamp_source_abs_delta_seconds": float(abs(signed_delta)),
        "frame_timestamp_source_warning": bool(
            abs(signed_delta) > SOURCE_DELTA_WARNING_SECONDS
        ),
    }
