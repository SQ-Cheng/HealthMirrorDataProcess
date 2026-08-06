"""Configuration for raw-video plus ECG Mamba abnormal-score regression."""

import os

from study.exp2_lab_multimodal.config import SEED
from study.exp2_face_pretrained_head32_regression.config import (
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SMOOTH_L1_BETA,
)


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.abspath(
    os.path.join(EXP_DIR, "..", "exp2_face_only", "outputs_aug20_24h")
)
RAW_DATA_ROOT = "/root/shared/HealthMirrorDataset"
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
LOG_DIR = os.path.join(EXP_DIR, "logs")

TARGETS = (
    "hemoglobin_low",
    "pco2_low",
    "po2_low",
    "high_blood_pressure",
    "lactate_high",
)
LAB_TARGET_PREFIXES = {
    "hemoglobin_low": "hemoglobin",
    "pco2_low": "pco2",
    "po2_low": "po2",
    "lactate_high": "lactate",
}

# Raw video and timestamp-based ECG resampling contract.
VIDEO_HEIGHT = 128
VIDEO_WIDTH = 128
WINDOW_SECONDS = 8.0
WINDOW_STRIDE_SECONDS = 8.0
MIN_VIDEO_FRAMES_PER_WINDOW = 64
MIN_ECG_SAMPLES_PER_WINDOW = 1024
ECG_SAMPLE_RATE_HZ = 512
ECG_SAMPLES_PER_WINDOW = int(WINDOW_SECONDS * ECG_SAMPLE_RATE_HZ)
ECG_MAX_INTERPOLATION_GAP_SECONDS = 0.060
ECG_CACHE_RECORDINGS = 4
TIMESTAMP_CACHE_RECORDINGS = 8

# Learned tokenization and official Mamba selective-SSM stack.
D_MODEL = 96
D_STATE = 16
D_CONV = 4
EXPAND = 2
MAMBA_LAYERS = 4
ECG_FIRST_STRIDE = 4
ECG_SECOND_STRIDE = 4
ECG_TOTAL_STRIDE = ECG_FIRST_STRIDE * ECG_SECOND_STRIDE
HEAD_HIDDEN_FEATURES = 32
DROPOUT = 0.10
HORIZONTAL_FLIP_PROBABILITY = 0.50

TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 24
TRAIN_NUM_WORKERS = 4
EVAL_NUM_WORKERS = 2
PREFETCH_FACTOR = 2

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
GRAD_CLIP_NORM = 1.0

MAMBA_SSM_VERSION = "2.2.6.post3"
TRANSFORMERS_VERSION = "4.44.2"
