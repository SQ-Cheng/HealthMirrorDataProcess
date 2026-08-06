"""Configuration for the Biosensors 2025 residual 3D CNN reproduction."""

import os


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
LOG_DIR = os.path.join(EXP_DIR, "logs")
REFERENCE_OUTPUT_DIR = os.path.abspath(
    os.path.join(EXP_DIR, "..", "exp2_face_pretrained_head32_regression", "outputs", "20frame")
)
ALLFRAME_INDEX_DIR = os.path.abspath(
    os.path.join(EXP_DIR, "..", "exp2_face_pretrained_head32_regression", "outputs", "allframes", "frame_index")
)

TARGET = "hemoglobin_low"
SEED = 20250317
FRAMES_PER_CLIP = 224
IMAGE_SIZE = 224
TRAIN_MICRO_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = 4
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_VAL_MSE = 0.3
GRAD_CLIP_NORM = 5.0
WEIGHT_DECAY = 0.0
COMPILE_MODEL = False

PAPER_DOI = "10.3390/bios15080485"
PAPER_MODEL = "Residual Regression Model"
