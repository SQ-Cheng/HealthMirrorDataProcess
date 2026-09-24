"""Run the matched change-direction analysis for the history-only model."""

from pathlib import Path

from study.exp2_face_history_head32_regression.analyze_change_direction import (
    BOOTSTRAP_REPLICATES,
    run,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
INPUT_DIR = EXPERIMENT_DIR / "outputs"
HISTORY_DIR = (
    EXPERIMENT_DIR.parent / "exp2_face_history_head32_regression/outputs/20frame"
)
OUTPUT_DIR = INPUT_DIR / "change_direction_analysis"


def main():
    run(
        input_dir=INPUT_DIR,
        history_dir=HISTORY_DIR,
        output_dir=OUTPUT_DIR,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        model_label="History-only Head32 regression",
        prediction_architecture=None,
    )


if __name__ == "__main__":
    main()
