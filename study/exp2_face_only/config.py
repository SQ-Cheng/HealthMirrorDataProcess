"""Configuration for Exp2 face-only lab/vitals prediction."""

import os

from study.exp2_lab_multimodal.config import (
    DATA_ROOT,
    FACE_FRAME_INDEX,
    FACE_SIZE,
    LAB_CSV,
    LAB_MATCH_MAX_DELTA_HOURS,
    PATIENT_INFO_GLOB,
    PLACEHOLDER_HOSPITAL_IDS,
    ROOT_DIR,
    SEED,
    TARGETS,
)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(EXP_DIR, "checkpoints")
LOG_DIR = os.path.join(EXP_DIR, "logs")

# Face-only model. Keep capacity modest because labels are sparse and patient N is small.
FACE_CHANNELS = [1, 16, 32, 64]
FACE_EMBED_DIM = 96
CLASSIFIER_HIDDEN = 64
DROPOUT = 0.45
AUGMENT_HORIZONTAL_FLIP = True

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 160
EARLY_STOPPING_PATIENCE = 30
LR_SCHEDULER_PATIENCE = 12
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 1.0
MIN_TRAIN_SAMPLES_PER_CLASS = 8
POS_WEIGHT_MAX = 20.0
