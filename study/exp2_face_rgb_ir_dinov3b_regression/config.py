"""Six fusion structures, one verified paired cohort, and shared frozen features."""

from pathlib import Path

from study.exp2_face_dinov3_farl_regression import config as farl_config
from study.exp2_face_dinov3_frozen import run_main_regression as reference
from study.exp2_face_rgb_ir_head32_regression import config as paired

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
CACHE = HERE / "cache"
TARGETS = paired.TARGETS
GROUPS = ("dino", "dino_farl")
METHODS = ("concat", "vector_gate", "scalar_gate")
VARIANTS = tuple(f"{group}_{method}" for group in GROUPS for method in METHODS)
BRANCHES = ("rgb_dino", "ir_dino", "rgb_farl", "ir_farl")
DIMENSIONS = {"rgb_dino": 768, "ir_dino": 768, "rgb_farl": 512, "ir_farl": 512}
VIEWS = tuple(reference.config.VIEWS)
RGB_INDEX = paired.CACHE / "rgb.npz"
IR_INDEX = paired.CACHE / "ir.npz"


def branches_for(architecture):
    if architecture not in VARIANTS:
        raise ValueError(architecture)
    return (("rgb_dino", "rgb_farl", "ir_dino", "ir_farl")
            if architecture.startswith("dino_farl_") else ("rgb_dino", "ir_dino"))


def method_for(architecture):
    return architecture.removeprefix("dino_farl_") if architecture.startswith("dino_farl_") else architecture.removeprefix("dino_")
