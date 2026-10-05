"""R2 and explained-variance analysis for face-only regression variants."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from study.exp2_face_history_head32_regression import (
    plot_r2_explained_variance as analysis,
)


EXP_DIR = Path(__file__).resolve().parent
analysis.EXP_DIR = EXP_DIR
analysis.OUTPUT_ROOT = EXP_DIR / "outputs"
analysis.VARIANTS = {
    "20frame_face224": "Native 224 / 20 frames",
}
analysis.TARGETS = {
    "hemoglobin_low": "Hemoglobin",
    "lactate_high": "Lactate",
    "oxyhemoglobin_fraction": "Oxyhemoglobin fraction",
    "urea_high": "Urea",
    "total_bilirubin_high": "Total bilirubin",
    "platelet_count_low": "Platelet count",
    "aa_po2_ratio_low": "A/a PO2 ratio",
    "creatinine_high": "Creatinine",
}
analysis.COLORS = {"Native 224 / 20 frames": "#4C78A8"}
analysis.FIGURE_TITLE = "Face-only regression: video-level test goodness of fit"


if __name__ == "__main__":
    analysis.main()
