"""Build labels with the exact source pipeline used by the history control."""

from study.exp2_face_history_head32_regression.source_data import (
    LAB_SOURCE_DEFINITIONS,
    TARGET_ANALYTES,
    build_raw_video_source,
    validate_analyte_source_policies,
)


__all__ = (
    "LAB_SOURCE_DEFINITIONS",
    "TARGET_ANALYTES",
    "build_raw_video_source",
    "validate_analyte_source_policies",
)
