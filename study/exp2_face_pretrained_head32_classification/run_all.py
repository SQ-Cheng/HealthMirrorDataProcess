"""Run the 24h frame-loss face classifier against the shared regression cohort."""

from study.common.run_face_main_24h import main


if __name__ == "__main__":
    main(default_families=("classification",))
