"""Optional result namespace for a refreshed laboratory-data cohort."""

import os
from pathlib import Path
import re


STUDY = Path(__file__).resolve().parents[1]
RUN_TAG = os.environ.get("HEALTHMIRROR_LAB_RUN_TAG", "")
OVERWRITE = os.environ.get("HEALTHMIRROR_LAB_OVERWRITE", "0") == "1"
if RUN_TAG and not re.fullmatch(r"lab_update_[0-9]{8}(?:_[a-z0-9]+)?", RUN_TAG):
    raise ValueError("Invalid HEALTHMIRROR_LAB_RUN_TAG")


def versioned(path):
    path = Path(path)
    return path / RUN_TAG if RUN_TAG and not OVERWRITE else path
