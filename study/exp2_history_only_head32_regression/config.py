"""Configuration for the controlled history-only regression ablation."""

from pathlib import Path

from study.exp2_face_history_head32_regression.config import (
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    HEAD_HIDDEN_FEATURES,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    HISTORY_HIDDEN_FEATURES,
    HISTORY_INPUT_FEATURES,
    HISTORY_OUTPUT_FEATURES,
    MIN_LEARNING_RATE,
    SCORE_DEFINITIONS,
    SEED,
    SMOOTH_L1_BETA,
    TARGETS,
    WEIGHT_DECAY,
)


EXP_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = (
    EXP_DIR.parent / "exp2_face_history_head32_regression" / "outputs" / "20frame"
).resolve()
FACE_ONLY_REFERENCE_DIR = (
    EXP_DIR.parent / "exp2_face_pretrained_head32_regression" / "outputs" / "20frame"
).resolve()
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_DIR = EXP_DIR / "logs"

MODEL_NAME = "history_only_head32"
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0
