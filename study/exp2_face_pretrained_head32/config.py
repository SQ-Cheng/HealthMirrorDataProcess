"""Configuration for pretrained single-task face classification experiments."""

import os

from study.exp2_lab_multimodal.config import SEED, TARGETS


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.abspath(
    os.path.join(EXP_DIR, "..", "exp2_face_only", "outputs_aug20_24h")
)
WEIGHTS_DIR = os.path.abspath(
    os.path.join(EXP_DIR, "..", "exp2_face_pretrained", "pretrained_weights")
)
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
LOG_DIR = os.path.join(EXP_DIR, "logs")

ARCHITECTURES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")
HEAD_HIDDEN_FEATURES = 32
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VIEW_NAMES = ("original", "hflip", "center_crop", "brightness", "contrast")
CROP_SCALE = 0.90
BRIGHTNESS_DELTA = 0.06
CONTRAST_DELTA = 0.08

BATCH_SIZE = 32
NUM_WORKERS = 2
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
HEAD_MAX_EPOCHS = 40
FINETUNE_MAX_EPOCHS = 60
HEAD_PATIENCE = 10
FINETUNE_PATIENCE = 12
MIN_LEARNING_RATE = 1e-6
GRAD_CLIP_NORM = 1.0

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20
MIN_VIDEOS_PER_CLASS = 5
MIN_PATIENTS_PER_CLASS = 3
POS_WEIGHT_MAX = 15.0
POS_WEIGHT_MIN = 1.0 / POS_WEIGHT_MAX
