#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGRESSION="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
echo "[monitor-start] waiting for exp2_match_window_schedule_30_40"
while screen -ls 2>/dev/null | grep -q '[.]exp2_match_window_schedule_30_40[[:space:]]'; do
    sleep 30
done
for hours in 12 6; do
    previous="${REGRESSION}/outputs/ablations/lab_match_${hours}h_patient_diverse_schedule_30_40"
    if [[ ! -f "${previous}/COMPLETE" || ! -f "${previous}/figures/schedule_comparison.png" ]]; then
        echo "[monitor-error] predecessor ${hours}h regression ablation is incomplete" >&2
        exit 1
    fi
done
for hours in 12 6; do
    echo "[classification-start] matching window=${hours}h"
    "${PYTHON}" -u -m study.exp2_face_pretrained_head32_classification.run_match_window_ablation --hours "${hours}"
done
echo "[all-classification-window-ablations-complete]"
