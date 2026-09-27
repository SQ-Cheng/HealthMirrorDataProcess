#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
session=exp2_face_regression_warmup_monitor
output=study/exp2_face_pretrained_head32_regression/outputs/ablations/patient_diverse_schedule_30_40_warmup2
log=study/exp2_face_pretrained_head32_regression/logs/patient_diverse_schedule_30_40_warmup2/run.log

if screen -ls | grep -Fq ".$session"; then
  echo "Screen session already exists: $session" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "Output exists; refusing to overwrite: $output" >&2
  exit 1
fi
mkdir -p "$(dirname "$log")"
screen -L -Logfile "$PWD/$log" -dmS "$session" bash -lc \
  "cd '$PWD' && exec bash study/exp2_face_pretrained_head32_regression/monitor_binary_then_warmup.sh"
echo "Detached monitor: $session"
echo "Log: $PWD/$log"
