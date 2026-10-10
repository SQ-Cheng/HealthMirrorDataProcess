"""Independent DINO experiment; clinical data remain owned by main Exp6."""

from pathlib import Path

from study.exp6_face_pair_lab_delta import config as baseline
from study.exp2_face_dinov3_frozen import config as dino


HERE = Path(__file__).resolve().parent
BASELINE = baseline.NATIVE_OUTPUT_DIR
OUTPUT = HERE / "outputs"
CACHE = HERE / "cache/cls_features"
HEAD_WIDTHS = (32, 64)
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-3
SEED = baseline.SEED
VIEWS = baseline.VIEWS
FRAME_PAIR_BATCH_SIZE = 240
MICROBATCH_FRAME_PAIRS = 120
GRAD_CLIP = 1.0
