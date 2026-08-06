#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
SESSION_NAME="${SESSION_NAME:-exp2_face_history_head32_regression_20frame}"
LOG_DIR="${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/20frame"
LOG_FILE="${LOG_DIR}/run.log"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
EXTRA_ARGS=""
LOG_MODE="overwrite"
if (( $# > 0 )); then
    printf -v EXTRA_ARGS ' %q' "$@"
    for argument in "$@"; do
        if [[ "${argument}" == "--replace-targets" ]]; then
            LOG_MODE="append"
        fi
    done
fi
if [[ "${LOG_MODE}" == "append" ]]; then
    TEE_ARGS="-a"
    printf '\n[add-target-start] %s args=%q\n' "$(date -Is)" "$*" >> "${LOG_FILE}"
else
    TEE_ARGS=""
fi
COMMAND="set -o pipefail; cd ${ROOT_DIR} && source /root/miniconda3/etc/profile.d/conda.sh && conda activate healthmirrorenv && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 python -u -m study.exp2_face_history_head32_regression.run_all${EXTRA_ARGS} 2>&1 | tee ${TEE_ARGS} ${LOG_FILE}"
screen -dmS "${SESSION_NAME}" bash -lc "${COMMAND}"
echo "Started detached screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
