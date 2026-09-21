#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
SESSION_NAME="${SESSION_NAME:-exp2_eight_target_regression_chain}"
RUN_SCRIPT="${ROOT_DIR}/study/exp2_face_history_head32_regression/run_eight_target_chain.sh"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

screen -dmS "${SESSION_NAME}" bash -lc "exec bash ${RUN_SCRIPT}"
echo "Started detached screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Chain log: ${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/eight_target_chain.log"
