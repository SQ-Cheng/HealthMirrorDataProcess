#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
SESSION_NAME="${SESSION_NAME:-exp2_face_pretrained}"
LOG_DIR="${ROOT_DIR}/study/exp2_face_pretrained/logs"
LOG_FILE="${LOG_DIR}/run.log"

if screen -ls | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
EXTRA_ARGS=""
if (( $# > 0 )); then
    printf -v EXTRA_ARGS ' %q' "$@"
fi
COMMAND="cd ${ROOT_DIR} && source /root/miniconda3/etc/profile.d/conda.sh && conda activate healthmirrorenv && MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 python -u -m study.exp2_face_pretrained.run_all${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}"
screen -dmS "${SESSION_NAME}" bash -lc "${COMMAND}"

echo "Started detached screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
