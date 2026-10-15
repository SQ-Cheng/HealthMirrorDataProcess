"""Same current 24h DINO clinical/training protocol, three independent heads."""

from pathlib import Path
from study.exp2_face_dinov3_frozen import run_main_regression as reference


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
CACHE = HERE / "cache/farl512"
WEIGHTS = HERE.parent / "common/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
DINO_CACHE = reference.CACHE
INDEX_PATH = reference.INDEX_PATH
TARGETS = reference.TARGETS
BASE_VARIANTS = ("concat", "gated")
VARIANTS = BASE_VARIANTS + ("scalar_gated",)
CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_MEAN = (.48145466, .4578275, .40821073)
CLIP_STD = (.26862954, .26130258, .27577711)
DINO_BACKBONE = "dinov3_vits16"
DINO_FEATURES = 384


def configure_backbone(backbone="dinov3_vits16"):
    global OUTPUT, DINO_CACHE, DINO_BACKBONE, DINO_FEATURES
    if backbone not in ("dinov3_vits16", "dinov3_vitb16"):
        raise ValueError("Unsupported DINO backbone")
    reference.configure_variant(32, backbone)
    DINO_BACKBONE = backbone
    DINO_FEATURES = reference.FEATURE_DIM
    DINO_CACHE = reference.CACHE
    OUTPUT = HERE / ("outputs/vitb16" if backbone == "dinov3_vitb16" else "outputs")
