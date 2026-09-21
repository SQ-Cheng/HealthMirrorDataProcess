#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
SESSION_NAME="${SESSION_NAME:-monitor_history_only_after_face_only}"
SCRIPT="$ROOT/study/exp2_history_only_head32_regression/monitor_pretrained_then_launch.sh"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
screen -dmS "$SESSION_NAME" bash -lc "bash '$SCRIPT'"
echo "Started detached monitor screen: $SESSION_NAME"
echo "Monitor log: $ROOT/study/exp2_history_only_head32_regression/logs/monitor_pretrained_then_launch.log"
