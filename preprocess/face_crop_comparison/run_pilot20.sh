#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p preprocess/face_crop_comparison/logs
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
/root/miniconda3/envs/healthmirrorenv/bin/python -u \
    -m preprocess.face_crop_comparison.run_alignment_comparison "$@" \
    2>&1 | tee preprocess/face_crop_comparison/logs/mediapipe_kalman_alignment20.log
