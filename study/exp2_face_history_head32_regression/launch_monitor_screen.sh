#!/usr/bin/env bash
set -euo pipefail

root="/root/autodl-tmp/HealthMirrorDataProcess"
session="monitor_exp5_then_exp2_history_full"
cd "$root"
if screen -list 2>/dev/null | grep -q "[.]${session}[[:space:]]"; then
  echo "screen session already exists: ${session}" >&2
  exit 1
fi
screen -dmS "$session" bash -lc \
  "cd '$root' && bash study/exp2_face_history_head32_regression/monitor_exp5_then_launch_full.sh"
echo "Started detached monitor: ${session}"
