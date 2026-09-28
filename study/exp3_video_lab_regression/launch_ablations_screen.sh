#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
experiment=study/exp3_video_lab_regression
session=exp3_video_lab_ablations
log="$experiment/logs/ablations/run.log"

if screen -ls | grep -Fq ".$session"; then
  echo "Screen session already exists: $session" >&2
  exit 1
fi
if [[ -e "$experiment/outputs/ablations/patient_diverse_schedule_30_40" \
   || -e "$experiment/outputs/ablations/middle48" ]]; then
  echo "Ablation output exists; refusing to overwrite" >&2
  exit 1
fi
mkdir -p "$(dirname "$log")"
screen -L -Logfile "$PWD/$log" -dmS "$session" bash -lc \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 exec /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp3_video_lab_regression.run_ablations"
echo "Detached screen session: $session"
echo "Log: $PWD/$log"
