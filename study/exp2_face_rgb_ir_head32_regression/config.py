"""Inherit the current primary clinical and two-stage regression protocol."""

from pathlib import Path

from study.exp2_face_pretrained_head32_regression.config import (
    ALL_REGRESSION_TARGETS, IMAGENET_MEAN, IMAGENET_STD, SCORE_DEFINITIONS,
    WEIGHTS_DIR, HEAD_MAX_EPOCHS, FINETUNE_MAX_EPOCHS,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = HERE.parent / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
PAIRED = ROOT / "preprocess/face_crop_comparison/outputs/rgb_ir_paired224"
CACHE = HERE / "cache/paired20"
OUTPUT = HERE / "outputs"
TARGETS = ALL_REGRESSION_TARGETS
MEAN = IMAGENET_MEAN + (.5,)
STD = IMAGENET_STD + (.5,)
