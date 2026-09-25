"""Configuration for robust-scaled raw-value face regression."""

import os

from study.exp2_lab_multimodal.config import DATA_ROOT, SEED
from study.exp2_face_history_head32_regression.config import (
    SCORE_DEFINITIONS,
    TARGETS,
)


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.abspath(
    os.path.join(EXP_DIR, "..", "common", "pretrained_weights")
)
OUTPUT_ROOT = os.path.join(EXP_DIR, "outputs")
OUTPUT_DIRS = {
    "20frame": os.path.join(OUTPUT_ROOT, "20frame"),
    "allframes": os.path.join(OUTPUT_ROOT, "allframes"),
}
OUTPUT_DIR = OUTPUT_DIRS["20frame"]
LOG_DIR = os.path.join(EXP_DIR, "logs")
REFERENCE_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        EXP_DIR,
        "..",
        "exp2_face_history_head32_regression",
        "outputs",
        "20frame",
    )
)
SOURCE_DATA_DIR = os.path.join(OUTPUT_DIR, "source_data")
REFERENCE_INDEX_DIR = os.path.abspath(
    os.path.join(
        EXP_DIR, "..", "exp2_face_history_head32_regression", "cache", "20frame_index"
    )
)

ARCHITECTURES = ("efficientnet_b0",)
HEAD_HIDDEN_FEATURES = 32
TORCH_COMPILE_ENABLED = True
TORCH_COMPILE_MODE = "reduce-overhead"

SOURCE_IMAGE_SIZE = 128
FRAMES_PER_VIDEO = 20
FRAME_QUANTILES = tuple(0.05 + 0.90 * index / 19 for index in range(20))
MIN_SOURCE_FRAME_GAP = 2
LAB_MATCH_MAX_DELTA_HOURS = 24.0
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIEW_NAMES = ("original", "hflip", "center_crop", "brightness", "contrast")
CROP_SCALE = 0.90
BRIGHTNESS_DELTA = 0.06
CONTRAST_DELTA = 0.08

REGRESSION_TARGET_COLUMN = "robust_scaled_raw_value"
REGRESSION_TARGET_TRANSFORM = "train_only_median_iqr"
# Retained for split-distribution auditing and legacy comparison utilities.
SCORE_TRANSFORM = "asinh"
SMOOTH_L1_BETA = 0.5

TRAIN_SOURCE_BATCH_SIZES = {
    "mobilenet_v3_small": 128,
    "efficientnet_b0": 48,
    "shufflenet_v2_x1_0": 48,
}
EVAL_BATCH_SIZES = {
    "mobilenet_v3_small": 1024,
    "efficientnet_b0": 512,
    "shufflenet_v2_x1_0": 512,
}
TRAIN_NUM_WORKERS = 6
EVAL_NUM_WORKERS = 2
PREFETCH_FACTOR = 4
FRAME_SHUFFLE_CHUNK_SIZE = 256
MAX_OPEN_FILES_PER_WORKER = 64
DECODE_CACHE_FRAMES = 16
JPEG_DECODER = "torchvision.io.decode_jpeg_cpu"

HEAD_LEARNING_RATE = 2e-4
FINETUNE_LEARNING_RATE = 1e-5
DIRECT_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
HEAD_MAX_EPOCHS = 40
FINETUNE_MAX_EPOCHS = 60
HEAD_PATIENCE = 10
FINETUNE_PATIENCE = 12
DIRECT_PATIENCE = 12
MIN_LEARNING_RATE = 1e-6
GRAD_CLIP_NORM = 1.0

MIN_VIDEOS_PER_CLASS = 5
MIN_PATIENTS_PER_CLASS = 3
POS_WEIGHT_MAX = 15.0
POS_WEIGHT_MIN = 1.0 / POS_WEIGHT_MAX

SPLIT_CANDIDATES = 512
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
SPLIT_KS_MAX = 0.20
SPLIT_WASSERSTEIN_IQR_MAX = 0.20
SPLIT_SMALL_N = 40
SPLIT_SMALL_KS_MAX = 0.30
SPLIT_SMALL_WASSERSTEIN_IQR_MAX = 0.25
SPLIT_SIZE_FRACTION_MAX = 0.05
SPLIT_POSITIVE_RATE_RANGE_MAX = 0.10
