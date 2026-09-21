"""Configuration layered on the controlled Head32 regression baseline."""

from pathlib import Path

from study.exp2_face_pretrained_head32_regression.config import *  # noqa: F403


EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
LOG_DIR = EXP_DIR / "logs"
REFERENCE_OUTPUT_DIR = EXP_DIR.parent / "exp2_face_pretrained_head32_regression/outputs/20frame"
REFERENCE_INDEX_PATH = (
    EXP_DIR.parent
    / "exp2_face_history_head32_regression/cache/20frame_index/frame_offsets.npz"
)

SIMCLR_EPOCHS = 20
SIMCLR_BATCH_SIZE = 64
SIMCLR_NUM_WORKERS = 4
SIMCLR_TEMPERATURE = 0.10
SIMCLR_PROJECTION_HIDDEN = 256
SIMCLR_PROJECTION_DIM = 128
SIMCLR_BACKBONE_LR = 1e-5
SIMCLR_PROJECTOR_LR = 2e-4
SIMCLR_MIN_LR_RATIO = 0.10
