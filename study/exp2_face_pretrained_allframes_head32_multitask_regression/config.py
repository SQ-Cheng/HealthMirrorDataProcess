"""Configuration for multi-output abnormal-score regression."""

import os

from study.exp2_face_pretrained_head32_regression.config import (
    BRIGHTNESS_DELTA,
    CONTRAST_DELTA,
    CROP_SCALE,
    DATA_ROOT,
    DECODE_CACHE_FRAMES,
    EVAL_BATCH_SIZES,
    EVAL_NUM_WORKERS,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_EPOCHS,
    FINETUNE_PATIENCE,
    FRAME_SHUFFLE_CHUNK_SIZE,
    GRAD_CLIP_NORM,
    HEAD_HIDDEN_FEATURES,
    HEAD_LEARNING_RATE,
    HEAD_MAX_EPOCHS,
    HEAD_PATIENCE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    JPEG_DECODER,
    MAX_OPEN_FILES_PER_WORKER,
    MIN_LEARNING_RATE,
    PREFETCH_FACTOR,
    SCORE_DEFINITIONS,
    SCORE_TRANSFORM,
    SEED,
    SMOOTH_L1_BETA,
    SOURCE_DATA_DIR,
    SOURCE_IMAGE_SIZE,
    SPLIT_FRACTIONS,
    SPLIT_KS_MAX,
    SPLIT_POSITIVE_RATE_RANGE_MAX,
    SPLIT_SIZE_FRACTION_MAX,
    SPLIT_SMALL_KS_MAX,
    SPLIT_SMALL_N,
    SPLIT_SMALL_WASSERSTEIN_IQR_MAX,
    SPLIT_WASSERSTEIN_IQR_MAX,
    TARGETS,
    TORCH_COMPILE_ENABLED,
    TORCH_COMPILE_MODE,
    TRAIN_NUM_WORKERS,
    TRAIN_SOURCE_BATCH_SIZES,
    VIEW_NAMES,
    WEIGHT_DECAY,
    WEIGHTS_DIR,
)


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
LOG_DIR = os.path.join(EXP_DIR, "logs")

ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
NUM_OUTPUTS = len(TARGETS)

# A global multi-label split has a smaller feasible region than five independent
# splits. Real-data validation found valid assignments within this search budget.
SPLIT_CANDIDATES = 2048
