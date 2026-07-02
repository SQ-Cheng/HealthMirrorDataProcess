"""Configuration for Exp2: Multi-Modal Deep Learning for Lab Test Prediction."""

import os

# ── Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = "/root/shared/HealthMirrorDataset"
LAB_CSV = os.path.join(ROOT_DIR, "merged_lab_tests.csv")
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(EXP_DIR, "checkpoints")
LOG_DIR = os.path.join(EXP_DIR, "logs")

# ── Random seed ───────────────────────────────────────────────────────
SEED = 20260702

# ── Signal parameters ─────────────────────────────────────────────────
ECG_LENGTH = 256          # resampled ECG length
ECG_WINDOW_SEC = 10.0     # ECG window duration in seconds

# ── Face / rPPG parameters ────────────────────────────────────────────
FACE_SIZE = 32            # face image resize (FACE_SIZE × FACE_SIZE)
FACE_FRAME_INDEX = 30     # which JPEG frame to extract from video

# ── Data filtering ────────────────────────────────────────────────────
PLACEHOLDER_HOSPITAL_IDS = {"", "-1", "1111111111", "1234567891", "nan", "None"}

# ── Lab test targets (15 binary classification tasks) ─────────────────
# Each is derived from lab test values using clinical thresholds.
TARGETS = [
    "lactate_high",
    "troponin_high",
    "glucose_high",
    "hemoglobin_low",
    "po2_low",
    "pco2_abnormal",
    "high_blood_pressure",
    "coronary_context",
    "lactate_moderate_high",
    "troponin_extreme_high",
    "glucose_marked_high",
    "hemoglobin_moderate_low",
    "po2_moderate_low",
    "pco2_low",
    "pco2_high",
]

# ── Model architecture ────────────────────────────────────────────────
# ECG encoder: lightweight 1D CNN
ECG_ENC_CHANNELS = [1, 16, 32, 64]            # conv layer channels (lightweight)
ECG_ENC_KERNELS = [7, 5, 5]                   # kernel sizes
ECG_ENC_STRIDES = [2, 2, 2]                   # strides
ECG_EMBED_DIM = 64                             # output embedding dim

# Face encoder: lightweight 2D CNN
FACE_ENC_CHANNELS = [1, 8, 16, 32]            # conv layer channels (lightweight)
FACE_EMBED_DIM = 64                            # output embedding dim

# Fusion + classifier
FUSION_HIDDEN = [128, 64]                      # [input_dim, hidden_dim]
DROPOUT = 0.5                                  # dropout rate in classifier (high for small data)

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 40                   # epochs without val improvement
LR_SCHEDULER_PATIENCE = 15                     # epochs for ReduceLROnPlateau
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 1.0

# ── Data split ────────────────────────────────────────────────────────
VAL_RATIO = 0.20
TEST_RATIO = 0.20                                # of total; val is VAL_RATIO of remaining

# ── Loss ──────────────────────────────────────────────────────────────
# Positive class weight (applied per-task to handle imbalance)
POS_WEIGHT = 2.0

# ── Multi-task masking ────────────────────────────────────────────────
# Tasks with < MIN_SAMPLES_PER_TASK positive or negative samples are skipped
MIN_SAMPLES_PER_TASK = 5
