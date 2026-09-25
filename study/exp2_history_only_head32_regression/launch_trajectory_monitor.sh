#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="exp2_history_trajectory_ablations"
LOG="${ROOT_DIR}/study/exp2_history_only_head32_regression/logs/trajectory_ablations.log"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen session already exists: ${SESSION}" >&2
    exit 1
fi
mkdir -p "$(dirname "${LOG}")"
screen -L -Logfile "${LOG}" -dmS "${SESSION}" bash \
    "${ROOT_DIR}/study/exp2_history_only_head32_regression/wait_views3_then_trajectory.sh"
echo "Started detached screen: ${SESSION}"
echo "Log: ${LOG}"
