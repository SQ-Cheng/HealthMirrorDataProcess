#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="exp2_face_reg_density_simclr"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
LOG_DIR="${EXP_DIR}/logs/density_simclr_ablations"
OUTPUT_ROOT="${EXP_DIR}/outputs/ablations"

for variant in patient_diverse_schedule_30_40_density_weighted patient_diverse_schedule_30_40_label_simclr; do
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
    "set -o pipefail; cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp2_face_pretrained_head32_regression.run_density_simclr_ablations 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached screen: ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/run.log"
