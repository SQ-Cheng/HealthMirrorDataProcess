#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_classification"
SESSION_NAME="exp2_classification_match_windows"
LOG_DIR="${EXP_DIR}/logs/ablations/match_windows"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
screen -dmS "${SESSION_NAME}" bash -lc \
    "set -o pipefail; bash '${EXP_DIR}/monitor_then_run_match_windows.sh' 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached screen: ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/run.log"
