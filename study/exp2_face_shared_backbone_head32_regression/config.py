"""Separate experiment paths, inheriting the matched main training settings."""

from pathlib import Path

from study.exp2_face_pretrained_head32_regression import config as reference
from study.exp2_face_pretrained_head32_regression.config import (
    ALL_REGRESSION_TARGETS, EVAL_NUM_WORKERS, FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS, FINETUNE_PATIENCE, GRAD_CLIP_NORM, HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS, HEAD_PATIENCE, MIN_LEARNING_RATE, PREFETCH_FACTOR,
    SCORE_DEFINITIONS, SEED, SMOOTH_L1_BETA, SPLIT_CANDIDATES,
    TRAIN_NUM_WORKERS, VIEW_NAMES, WEIGHTS_DIR, WEIGHT_DECAY,
)


EXP_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUT_DIR = Path(reference.OUTPUT_DIRS["20frame"])
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_DIR = EXP_DIR / "logs"
TORCH_COMPILE_ENABLED = True
TORCH_COMPILE_DYNAMIC = True
TORCH_COMPILE_CUDAGRAPHS = False
