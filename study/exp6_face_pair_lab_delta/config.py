"""Configuration for paired-face laboratory-value delta regression."""

from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
OUTPUT_DIR = EXP_DIR / "outputs"
CACHE_DIR = EXP_DIR / "cache" / "frames20"
LOG_DIR = EXP_DIR / "logs"
WEIGHTS_DIR = REPO_ROOT / "study/common/pretrained_weights"
PRETRAINED_WEIGHT_FILE = "efficientnet_b0_rwightman-7f5810bc.pth"

SEED = 20260808
TARGETS = (
    "oxyhemoglobin_fraction",
    "lactate_high",
    "urea_high",
    "troponin_high",
    "platelet_count_low",
    "hemoglobin_low",
    "aa_po2_ratio_low",
    "creatinine_high",
    "total_bilirubin_high",
)
TARGET_ANALYTES = {
    "oxyhemoglobin_fraction": "oxyhemoglobin_fraction",
    "lactate_high": "lactate",
    "urea_high": "urea",
    "troponin_high": "troponin",
    "platelet_count_low": "platelet_count",
    "hemoglobin_low": "hemoglobin",
    "aa_po2_ratio_low": "aa_po2_ratio",
    "creatinine_high": "creatinine",
    "total_bilirubin_high": "total_bilirubin",
}
TARGET_UNITS = {
    "oxyhemoglobin_fraction": "%",
    "lactate_high": "mmol/L",
    "urea_high": "mmol/L",
    "troponin_high": "ng/L",
    "platelet_count_low": "10^9/L",
    "hemoglobin_low": "g/L",
    "aa_po2_ratio_low": "%",
    "creatinine_high": "umol/L",
    "total_bilirubin_high": "umol/L",
}

FRAMES_PER_VIDEO = 20
SOURCE_IMAGE_SIZE = 128
IMAGE_SIZE = 224
VIEWS = ("original", "hflip", "center_crop", "brightness", "contrast")
CROP_SCALE = 0.90
BRIGHTNESS_DELTA = 0.06
CONTRAST_DELTA = 0.08
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HEAD_HIDDEN_FEATURES = 32

SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
SPLIT_CANDIDATES = 512
SPLIT_QUANTILE_BINS = 8

TRAIN_SOURCE_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 96
TRAIN_NUM_WORKERS = 6
EVAL_NUM_WORKERS = 2
PREFETCH_FACTOR = 3
MAX_OPEN_FILES_PER_WORKER = 64
DECODE_CACHE_FRAMES = 24

HEAD_LEARNING_RATE = 2e-4
FINETUNE_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
HEAD_MAX_EPOCHS = 40
FINETUNE_MAX_EPOCHS = 60
HEAD_PATIENCE = 10
FINETUNE_PATIENCE = 12
MIN_LEARNING_RATE = 1e-6
GRAD_CLIP_NORM = 1.0
SMOOTH_L1_BETA = 0.5
TORCH_COMPILE_ENABLED = True
TORCH_COMPILE_MODE = "reduce-overhead"
