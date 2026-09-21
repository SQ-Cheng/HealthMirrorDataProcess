"""Configuration for the CABG pre/post face-pair recovery experiment."""

from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
DATA_ROOT = Path("/root/shared/HealthMirrorDataset")
LAB_CSV = REPO_ROOT / "merged_lab_tests.csv"
WEIGHTS_DIR = REPO_ROOT / "study/exp2_face_pretrained/pretrained_weights"
PRETRAINED_WEIGHT_FILE = "efficientnet_b0_rwightman-7f5810bc.pth"
OUTPUT_DIR = EXP_DIR / "outputs"
CACHE_DIR = EXP_DIR / "cache" / "frames20"
LOG_DIR = EXP_DIR / "logs"

TIMEZONE = "Asia/Shanghai"
MAX_TIME_SOURCE_DELTA_SECONDS = 300.0
LAB_MATCH_MAX_HOURS = 24.0
SEED = 20260808
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
SPLIT_CANDIDATES = 512
SPLIT_SCORE_BINS = 8

ANALYTES = {
    "lactate": {
        "item": "乳酸浓度", "unit": "mmol/l", "valid_range": (0.1, 30.0),
    },
    "troponin": {
        "item": "肌钙蛋白Ⅰ", "unit": "ng/l",
        "valid_range": (0.0, 200000.0), "log1p": True,
    },
    "creatinine": {
        "item": "肌酐(Cr)测定", "unit": "μmol/l", "valid_range": (5.0, 2000.0),
    },
    "total_bilirubin": {
        "item": "总胆红素", "unit": "μmol/l", "valid_range": (0.1, 1000.0),
    },
    "platelet_count": {
        "item": "血小板", "unit": "10^9/l", "valid_range": (1.0, 2000.0),
    },
    "hemoglobin": {
        "item": "血红蛋白", "unit": "g/l", "valid_range": (20.0, 250.0),
    },
    "crp": {
        "item": "*快速C-反应蛋白", "unit": "mg/l", "valid_range": (0.0, 1000.0),
    },
    "albumin": {
        "item": "*白蛋白(Alb)测定-溴甲酚绿法", "unit": "g/l",
        "valid_range": (5.0, 80.0),
    },
    "po2": {
        "item": "氧分压", "unit": "mmhg", "valid_range": (10.0, 800.0),
    },
}
TRAJECTORY_BINS = 16
TRAJECTORY_GRID_SIZE = 201
TRAJECTORY_MIN_SCALE = 0.35
TARGET_COLUMN = "trajectory_deviation_score"

FRAMES_PER_VIDEO = 20
SOURCE_IMAGE_SIZE = 128
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_VIEWS = ("original", "hflip", "center_crop")
CROP_SCALE = 0.90
ENCODER_FEATURES = 64
HEAD_HIDDEN_FEATURES = 32

TRAIN_SOURCE_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 6
EVAL_NUM_WORKERS = 3
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
TORCH_COMPILE_MODE = "reduce-overhead"
