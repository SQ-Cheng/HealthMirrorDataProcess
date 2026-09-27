#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
experiment=study/exp2_face_pretrained_head32_classification
session=exp2_face_binary_patient_diverse_30_40
output="$experiment/outputs/ablations/patient_diverse_schedule_30_40"
log="$experiment/logs/patient_diverse_schedule_30_40/run.log"

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
  "cd '$PWD' && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 exec /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp2_face_pretrained_head32_classification.run_patient_diverse_schedule_ablation"
echo "Detached screen session: $session"
echo "Log: $PWD/$log"
