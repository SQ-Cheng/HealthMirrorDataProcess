#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
session=selected_exp2_exp3_5fold
log=study/exp2_face_pretrained_head32_regression/logs/selected_5fold/run.log

if screen -ls | grep -Fq ".$session"; then
  echo "Screen session already exists: $session" >&2
  exit 1
fi
mkdir -p "$(dirname "$log")"
screen -L -Logfile "$PWD/$log" -dmS "$session" bash -lc \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 exec /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.common.run_selected_5fold"
echo "Detached screen session: $session"
echo "Log: $PWD/$log"
