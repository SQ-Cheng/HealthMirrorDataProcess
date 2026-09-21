#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
SESSION_NAME="${SESSION_NAME:-exp2_remaining_three_pathway_chain}"
RUN_SCRIPT="${ROOT_DIR}/study/exp2_face_history_head32_regression/run_remaining_three_pathway_chain.sh"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

screen -dmS "${SESSION_NAME}" bash -lc "exec bash ${RUN_SCRIPT}"
echo "Started detached screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Log: ${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/remaining_three_pathway_chain.log"
