#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
experiment=study/exp6_face_pair_lab_delta_classifacation
session=exp6_pair_direction_classification
log="$experiment/logs/run.log"

if screen -ls | grep -Fq ".$session"; then
  echo "Screen session already exists: $session" >&2
  exit 1
fi
if [[ -e "$experiment/outputs" ]]; then
  echo "Output exists; refusing to overwrite: $experiment/outputs" >&2
  exit 1
fi
mkdir -p "$(dirname "$log")"
screen -L -Logfile "$PWD/$log" -dmS "$session" bash -lc \
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 exec /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp6_face_pair_lab_delta_classifacation.run_all"
echo "Detached screen session: $session"
echo "Log: $PWD/$log"
