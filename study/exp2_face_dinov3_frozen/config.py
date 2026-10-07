"""A single head-training stage on frozen DINOv3 ViT-S/16."""

from pathlib import Path

from study.common import run_video_loss_12h as reference
from study.common.run_distinct_lab_views_12h import OUTPUTS as BASELINES


HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "common/backbones/dinov3"
REPO_URL = "https://github.com/facebookresearch/dinov3.git"
REPO_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
WEIGHTS = HERE.parent / "common/pretrained_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
WEIGHT_HASH_PREFIX = "08c60483"
SOURCE = reference.SOURCE
INDEX_PATH = reference.INDEX_PATH
PREDECESSOR = HERE.parent / "exp2_face_architecture_ablation/outputs"
TARGETS = reference.config.TARGETS
VIEWS = reference.config.VIEW_NAMES
ARCHITECTURE = "dinov3_vits16_frozen"
FEATURES = 384
HIDDEN = 32
MAX_EPOCHS = 80
PATIENCE = 12
WEIGHT_DECAY = 1e-3
LEARNING_RATES = {ARCHITECTURE: 2e-4}
MIN_LEARNING_RATES = {ARCHITECTURE: 1e-6}
DROPOUT = .25
