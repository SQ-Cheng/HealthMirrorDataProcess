#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
WAIT_SESSION="${WAIT_SESSION:-exp2_face_history_head32}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_FILE="${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/20frame/hemoglobin_retrain.log"

session_exists() {
    screen -ls 2>/dev/null | grep -Eq "[.]${WAIT_SESSION}[[:space:]]"
}

echo "[monitor] waiting_for=${WAIT_SESSION} poll_seconds=${POLL_SECONDS}"
while session_exists; do
    sleep "${POLL_SECONDS}"
done

echo "[monitor] upstream session completed; starting hemoglobin replacement"
cd "${ROOT_DIR}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate healthmirrorenv
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1

python -u -m study.exp2_face_history_head32_regression.run_all \
    --targets hemoglobin_low \
    --replace-targets \
    2>&1 | tee "${LOG_FILE}"
