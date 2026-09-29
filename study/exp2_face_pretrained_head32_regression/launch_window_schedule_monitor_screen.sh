#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
SESSION_NAME="exp2_match_window_schedule_30_40"
LOG_DIR="${EXP_DIR}/logs/ablations/match_window_schedule_30_40"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
screen -dmS "${SESSION_NAME}" bash -lc \
    "set -o pipefail; bash '${EXP_DIR}/monitor_then_run_window_schedules.sh' 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached screen: ${SESSION_NAME}"
echo "Log: ${LOG_DIR}/run.log"
