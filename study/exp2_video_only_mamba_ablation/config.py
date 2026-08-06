"""Configuration inherited from the paired video+ECG Mamba experiment."""

import os

from study.exp2_video_ecg_mamba.config import (
    D_CONV,
    D_MODEL,
    D_STATE,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    EVAL_NUM_WORKERS,
    EXPAND,
    GRAD_CLIP_NORM,
    HEAD_HIDDEN_FEATURES,
    HORIZONTAL_FLIP_PROBABILITY,
    LEARNING_RATE,
    MAMBA_LAYERS,
    MAMBA_SSM_VERSION,
    MAX_EPOCHS,
    MIN_LEARNING_RATE,
    PREFETCH_FACTOR,
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    TARGETS,
    TIMESTAMP_CACHE_RECORDINGS,
    TRAIN_BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WEIGHT_DECAY,
    WINDOW_SECONDS,
)


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_EXP_DIR = os.path.abspath(os.path.join(EXP_DIR, "..", "exp2_video_ecg_mamba"))
PARENT_OUTPUT_DIR = os.path.join(PARENT_EXP_DIR, "outputs")
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
LOG_DIR = os.path.join(EXP_DIR, "logs")

# This exact token reproduces each paired multimodal job seed.
PAIRED_JOB_SEED_TOKEN = "video-ecg-mamba"
EXPECTED_PARENT_TASK_TYPE = "abnormal_score_regression"
EXPECTED_ECG_SAMPLE_RATE_HZ = 512
EXPECTED_ECG_MAX_GAP_SECONDS = 0.060
