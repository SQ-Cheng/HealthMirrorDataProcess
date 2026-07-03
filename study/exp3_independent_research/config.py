"""Shared configuration for Exp3: Independent Academic Research."""

import os

# ── Paths ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = "/root/shared/HealthMirrorDataset"
LAB_CSV = os.path.join(ROOT_DIR, "merged_lab_tests.csv")
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")

# ── Random seed ──
SEED = 20260703

# ── Data ──
ECG_LENGTH = 256
ECG_WINDOW_SEC = 10.0
FACE_SIZE = 32
PLACEHOLDER_HOSPITAL_IDS = {"", "-1", "1111111111", "1234567891", "nan", "None"}

# ── Lab analytes ──
ANALYTES = ["lactate", "troponin", "glucose", "hemoglobin", "po2", "pco2"]

# ── Ensure output directories exist ──
for _d in ["direction_a", "direction_b", "direction_c", "direction_d", "direction_e"]:
    os.makedirs(os.path.join(OUTPUT_DIR, _d), exist_ok=True)
