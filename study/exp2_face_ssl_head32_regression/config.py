"""Independent SSL protocol; downstream settings come from the face-only main."""

from pathlib import Path
from study.exp2_face_pretrained_head32_regression import config as reference


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
BASE = Path(reference.OUTPUT_DIRS["20frame"])
OUTPUT = EXP_DIR / "outputs"
LOGS = EXP_DIR / "logs"
CACHE = EXP_DIR / "cache"
TARGETS = reference.ALL_REGRESSION_TARGETS
SEED = reference.SEED
VIEWS = reference.VIEW_NAMES

SSL_EPOCHS = 8
SSL_WARMUP_EPOCHS = 2
SSL_ANCHORS_PER_GPU = 64
SSL_BACKBONE_LR = 1e-4
SSL_MLP_LR = 1e-3
SSL_MIN_LR = 1e-6
SSL_WEIGHT_DECAY = 1e-4
SSL_EMA_BASE = .996
SSL_MLP_HIDDEN = 2048
SSL_PROJECTION_DIM = 256
SSL_WORKERS = 8
SSL_PREFETCH = 2
SSL_LOG_EVERY = 100
GRAD_CLIP = 1.0
