"""Configuration for the native-resolution 24-hour Aug20 experiment."""

import os

from study.exp2_lab_multimodal.config import (
    DATA_ROOT,
    LAB_CSV,
    SEED,
    TARGETS,
)

EXPERIMENT_TARGETS = ("hemoglobin_low",)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# Keep the completed single-frame experiment intact.
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs_aug20_24h")
CHECKPOINT_DIR = os.path.join(EXP_DIR, "checkpoints_aug20_24h")
LOG_DIR = os.path.join(EXP_DIR, "logs_aug20_24h")

# Restrict the new augmentation experiment to one calendar day.
LAB_MATCH_MAX_DELTA_HOURS = 24.0

# Each matched event is expanded to one sample per selected frame.
NUM_FACE_FRAMES = 20
# Source MJPEG frames are 128x128. The builder validates this and never resizes them.
FACE_SIZE = 128
FRAME_QUANTILES = tuple(0.05 + 0.90 * index / 19 for index in range(20))
MIN_SOURCE_FRAME_GAP = 2

# Compact RGB residual classifier. One independent model is trained per task.
STEM_CHANNELS = 24
STAGE_CHANNELS = (24, 48, 80, 128)
STAGE_BLOCKS = (1, 2, 2, 2)
FACE_EMBED_DIM = 128
CLASSIFIER_HIDDEN = 96
DROPOUT = 0.30

BATCH_SIZE = 24
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 2e-3
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 24
GRAD_CLIP_NORM = 1.0
MIN_TRAIN_SAMPLES_PER_CLASS = 8
MIN_TRAIN_PATIENTS_PER_CLASS = 3
POS_WEIGHT_MAX = 15.0
NUM_WORKERS = 4

# Mild single-frame augmentation preserves medically relevant color cues.
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_CROP_MIN_SCALE = 0.90
AUGMENT_BRIGHTNESS = 0.06
AUGMENT_CONTRAST = 0.08
