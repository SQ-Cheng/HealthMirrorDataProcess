#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp2_face_history_head32_classification"
WATCH_SESSION="exp5_face_pair_recovery"
SESSION="exp2_binary_monitor"

if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$EXP/logs"
screen -dmS "$SESSION" bash -lc \
  "while screen -list | grep -q '[.]${WATCH_SESSION}[[:space:]]'; do sleep 60; done; \
   cd '$ROOT'; bash '$EXP/launch_screen.sh'"
echo "monitor started detached: $SESSION"
