from pathlib import Path

from study.exp2_face_rgb_ir_head32_regression import config as paired

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
TARGETS = paired.TARGETS
MICROBATCH = 60
RGB_INDEX = paired.CACHE / "rgb.npz"
IR_INDEX = paired.CACHE / "ir.npz"
