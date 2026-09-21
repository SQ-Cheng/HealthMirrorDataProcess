#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp2_face_pretrained_head32_regression_simclr"
SESSION="exp2_simclr_monitor"

if screen -list 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$EXP/logs"
screen -dmS "$SESSION" bash -lc \
  "exec bash '$EXP/run_after_binary.sh' > '$EXP/logs/monitor.log' 2>&1"
echo "monitor started detached: $SESSION"
echo "attach with: screen -r $SESSION"
echo "monitor log: $EXP/logs/monitor.log"
