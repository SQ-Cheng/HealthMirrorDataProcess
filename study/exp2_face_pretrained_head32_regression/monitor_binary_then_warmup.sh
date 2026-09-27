#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
session=exp2_face_binary_patient_diverse_30_40
binary_complete=study/exp2_face_pretrained_head32_classification/outputs/ablations/patient_diverse_schedule_30_40/COMPLETE

while screen -ls 2>/dev/null | grep -Eq "[.]${session}[[:space:]]"; do
  sleep 30
done
if [[ ! -f "$binary_complete" ]]; then
  echo "[not-started] Binary ablation did not complete: $binary_complete" >&2
  exit 1
fi

echo "[launch] Binary ablation complete; starting regression warmup ablation"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 \
  /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp2_face_pretrained_head32_regression.run_patient_diverse_schedule_warmup_ablation
