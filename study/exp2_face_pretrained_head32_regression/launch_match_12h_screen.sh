#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
HOURS="${1:-12}"
if [[ "${HOURS}" != "6" && "${HOURS}" != "12" ]]; then
    echo "Usage: $0 [6|12]" >&2
    exit 2
fi
OUTPUT_DIR="${EXP_DIR}/outputs/ablations/lab_match_${HOURS}h"
LOG_DIR="${EXP_DIR}/logs/ablations/lab_match_${HOURS}h"
INDEX_DIR="${ROOT_DIR}/study/exp2_face_history_head32_regression/cache/20frame_index"
SESSION_NAME="exp2_face_regression_lab_match_${HOURS}h"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Ablation output already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi
if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
screen -dmS "${SESSION_NAME}" bash -lc \
    "set -o pipefail; cd '${ROOT_DIR}'; export CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1; ( '${PYTHON}' -u -m study.exp2_face_pretrained_head32_regression.run_all --frame-policy 20frame --match-max-delta-hours '${HOURS}' --output-dir '${OUTPUT_DIR}' --source-dir '${OUTPUT_DIR}/source_data' --index-dir '${INDEX_DIR}' --reference-output-dir '' && '${PYTHON}' -u -m study.exp2_face_pretrained_head32_regression.plot_match_window_comparison --candidate-dir '${OUTPUT_DIR}' --candidate-hours '${HOURS}' ) 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached screen: ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/run.log"
