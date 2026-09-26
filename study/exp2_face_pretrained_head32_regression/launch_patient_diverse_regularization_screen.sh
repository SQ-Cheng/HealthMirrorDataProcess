#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="exp2_face_regression_regularization"
LOG_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/logs/patient_diverse_regularization"
OUTPUT_ROOT="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/outputs/ablations"

for variant in patient_diverse_schedule_30_40_wd1e2 patient_diverse_schedule_30_40_bn_fixed; do
    if [[ -e "${OUTPUT_ROOT}/${variant}" ]]; then
        echo "Ablation output already exists: ${OUTPUT_ROOT}/${variant}" >&2
        exit 1
    fi
done
if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
screen -dmS "${SESSION_NAME}" bash -lc \
    "set -o pipefail; cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp2_face_pretrained_head32_regression.run_patient_diverse_regularization_ablations 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached screen: ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/run.log"
