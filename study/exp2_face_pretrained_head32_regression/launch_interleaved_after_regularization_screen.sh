#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_NAME="exp2_face_regression_interleaved_views"
LOG_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/logs/interleaved_views"
OUTPUT_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/outputs/ablations/patient_diverse_schedule_30_40_interleaved_views"

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
    "set -o pipefail; bash '${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/monitor_regularization_then_interleaved.sh' 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached monitor screen: ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/run.log"
