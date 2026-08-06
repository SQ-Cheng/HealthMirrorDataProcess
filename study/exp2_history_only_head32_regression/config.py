"""Configuration for the controlled history-only regression ablation."""

from pathlib import Path

from study.exp2_face_history_head32_regression.config import (
    HEAD_HIDDEN_FEATURES,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    HISTORY_HIDDEN_FEATURES,
    HISTORY_INPUT_FEATURES,
    HISTORY_OUTPUT_FEATURES,
    MIN_LEARNING_RATE,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    WEIGHT_DECAY,
)


EXP_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = (
    EXP_DIR.parent / "exp2_face_history_head32_regression" / "outputs" / "20frame"
).resolve()
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_DIR = EXP_DIR / "logs"

TARGETS = ("hemoglobin_low", "po2_low")
MODEL_NAME = "history_only_head32"
BATCH_SIZE = 128
MAX_EPOCHS = HEAD_MAX_EPOCHS
PATIENCE = HEAD_PATIENCE
LEARNING_RATE = HEAD_LEARNING_RATE
GRAD_CLIP_NORM = 1.0
