#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
EXP_DIR="${ROOT_DIR}/study/exp2_face_history_head32_regression"
BASELINE_DIR="${EXP_DIR}/outputs/20frame"
OUTPUT_DIR="${EXP_DIR}/outputs/20frame_range_weighted"
REFERENCE_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/outputs/20frame"
INDEX_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/cache/20frame_index"
SESSION_NAME="${SESSION_NAME:-exp2_face_history_head32_range_weighted}"
LOG_FILE="${EXP_DIR}/logs/20frame_range_weighted/run.log"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}/run_index.csv" ]]; then
    echo "weighted output already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi
mkdir -p "$(dirname "${LOG_FILE}")"
COMMAND="set -o pipefail; cd ${ROOT_DIR} && source /root/miniconda3/etc/profile.d/conda.sh && conda activate healthmirrorenv && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 python -u -m study.exp2_face_history_head32_regression.run_all --targets hemoglobin_low,oxyhemoglobin_fraction --range-weighted --output-dir ${OUTPUT_DIR} --source-dir ${BASELINE_DIR}/source_data --reference-output-dir ${REFERENCE_DIR} --index-dir ${INDEX_DIR} 2>&1 | tee ${LOG_FILE} && python -u -m study.exp2_face_history_head32_regression.plot_range_weight_comparison --baseline-dir ${BASELINE_DIR} --weighted-dir ${OUTPUT_DIR} 2>&1 | tee -a ${LOG_FILE}"
screen -dmS "${SESSION_NAME}" bash -lc "${COMMAND}"
echo "Started detached screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
