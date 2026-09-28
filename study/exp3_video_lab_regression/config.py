"""Exp3 video-to-laboratory-value regression configuration."""

from pathlib import Path

from study.exp2_face_pretrained_head32_regression.config import SEED, TARGETS


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
SOURCE_DIR = ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame"
INDEX_DIR = EXP_DIR / "cache/clip16_index"
OUTPUT_DIR = EXP_DIR / "outputs"
WEIGHT_PATH = ROOT / "study/common/pretrained_weights/r3d_18-b3b3357e.pth"

SOURCE_SIZE = 128
CLIP_FRAMES = 16
CLIP_POSITIONS = (0.1, 0.9, 0.5)
CROP_SIZE = 112
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
HEAD_HIDDEN = 32

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 6
TRAIN_WORKERS = 4
EVAL_WORKERS = 2
PREFETCH_FACTOR = 2
MAX_OPEN_FILES_PER_WORKER = 24

HEAD_LR = 1e-3
HEAD_MIN_LR = 1e-5
HEAD_EPOCHS = 10
HEAD_PATIENCE = 4
FINETUNE_LR = 1e-5
FINETUNE_MIN_LR = 1e-7
FINETUNE_EPOCHS = 20
FINETUNE_PATIENCE = 6
WEIGHT_DECAY = 1e-4
SMOOTH_L1_BETA = 0.5
GRAD_CLIP_NORM = 1.0
