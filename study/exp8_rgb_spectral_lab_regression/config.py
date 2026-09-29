"""Paths and fixed protocol for the three-target spectral experiment."""

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
VARIANT = os.environ.get("EXP8_VARIANT", "ntire2022")
if VARIANT not in {"ntire2022", "hyperskin"}:
    raise ValueError(f"Unsupported Exp8 variant: {VARIANT}")
GRID_SIZE = int(os.environ.get("EXP8_GRID_SIZE", "4"))
if GRID_SIZE not in {4, 16}:
    raise ValueError(f"Unsupported Exp8 spatial grid: {GRID_SIZE}")
RUN_SUFFIX = ("_hyperskin" if VARIANT == "hyperskin" else "") + (
    f"_grid{GRID_SIZE}" if GRID_SIZE != 4 else ""
)
BASE = HERE.parent / "exp2_face_pretrained_head32_regression"
BASE_OUTPUT = BASE / "outputs" / "20frame"
INDEX_PATH = HERE.parent / "exp2_face_history_head32_regression" / "cache" / "20frame_index" / "frame_offsets.npz"
WEIGHT_PATH = HERE.parent / "common" / "pretrained_weights" / (
    "mst_plus_plus_hyperskin_rgb_vis.pth" if VARIANT == "hyperskin"
    else "mst_plus_plus_ntire2022.pth"
)
CACHE = HERE / f"cache{RUN_SUFFIX}"
OUTPUT = HERE / f"outputs{RUN_SUFFIX}"

TARGETS = ("hemoglobin_low", "total_bilirubin_high", "lactate_high")
TARGET_LABELS = {
    "hemoglobin_low": ("Hemoglobin", "g/L"),
    "total_bilirubin_high": ("Total bilirubin", "umol/L"),
    "lactate_high": ("Lactate", "mmol/L"),
}
WAVELENGTH_NM = tuple(range(400, 701, 10))
FEATURE_SHAPE = (31, GRID_SIZE, GRID_SIZE)
SOURCE_IMAGE_SIZE = 128
FRAMES_PER_VIDEO = 20
SEED = 42

HEAD_EPOCHS = 160
PATIENCE = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
SMOOTH_L1_BETA = 0.5
