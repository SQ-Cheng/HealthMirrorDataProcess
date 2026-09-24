"""Configuration for paired-face partial-CASUS regression."""

from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
DATA_ROOT = Path("/root/shared/HealthMirrorDataset")
LAB_CSV = REPO_ROOT / "merged_lab_tests.csv"
WEIGHTS_DIR = REPO_ROOT / "study/common/pretrained_weights"
PRETRAINED_WEIGHT_FILE = "efficientnet_b0_rwightman-7f5810bc.pth"
OUTPUT_DIR = EXP_DIR / "outputs"
CACHE_DIR = EXP_DIR / "cache" / "frames20"
POST_ONLY_OUTPUT_DIR = OUTPUT_DIR / "protocols" / "post_face_only"
POST_ONLY_CACHE_DIR = EXP_DIR / "cache" / "post_face_only_frames20"
LOG_DIR = EXP_DIR / "logs"

TIMEZONE = "Asia/Shanghai"
MAX_TIME_SOURCE_DELTA_SECONDS = 300.0
LAB_MATCH_MAX_HOURS = 24.0
SEED = 20260808
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
SPLIT_CANDIDATES = 512
SPLIT_SCORE_BINS = 8

# Exact aliases deliberately exclude eGFR, urine creatinine, LDH, platelet
# function tests, and direct/indirect bilirubin.
CASUS_ANALYTES = {
    "creatinine": {
        "items": ("*肌酐(Cr)测定", "*肌酐(Cr)测定-苦味酸法", "*肌酐"),
        "specimens": ("血", "血清", "血浆"),
        "valid_range": (0.1, 20.0),
        "canonical_unit": "mg/dL",
    },
    "bilirubin": {
        "items": ("*总胆红素(T-Bil)测定", "总胆红素(T-Bil)测定", "*总胆红素", "总胆红素"),
        "specimens": ("血", "血清", "血浆"),
        "valid_range": (0.05, 100.0),
        "canonical_unit": "mg/dL",
    },
    "lactate": {
        "items": ("*乳酸浓度", "乳酸浓度", "乳酸"),
        "specimens": ("血", "全血", "动脉血即刻", "即刻动脉血", "静脉血"),
        "valid_range": (0.1, 30.0),
        "canonical_unit": "mmol/L",
    },
    "platelets": {
        "items": ("*血小板", "血小板"),
        "specimens": ("血", "全血"),
        "valid_range": (1.0, 2000.0),
        "canonical_unit": "x10^3/uL",
    },
}
CASUS_MAX_SCORE = 16.0

FRAMES_PER_VIDEO = 20
SOURCE_IMAGE_SIZE = 128
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_VIEWS = ("original", "hflip", "center_crop")
CROP_SCALE = 0.90
ENCODER_FEATURES = 64
HEAD_HIDDEN_FEATURES = 32

# This is the global source-video batch. DataParallel divides it across GPUs.
TRAIN_SOURCE_BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
TRAIN_NUM_WORKERS = 12
EVAL_NUM_WORKERS = 8
PREFETCH_FACTOR = 3
MAX_OPEN_FILES_PER_WORKER = 64
DECODE_CACHE_FRAMES = 24

HEAD_LEARNING_RATE = 2e-4
FINETUNE_HEAD_LEARNING_RATE = 1e-4
FINETUNE_BACKBONE_LEARNING_RATE = 1e-5
HEAD_MAX_EPOCHS = 30
FINETUNE_MAX_EPOCHS = 50
HEAD_PATIENCE = 8
FINETUNE_PATIENCE = 10
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SMOOTH_L1_BETA = 0.10
TORCH_COMPILE_ENABLED = False
