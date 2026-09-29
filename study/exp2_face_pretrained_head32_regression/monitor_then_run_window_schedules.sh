#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
BASELINE_DIR="${EXP_DIR}/outputs/ablations/lab_match_6h"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
echo "[monitor-start] waiting for exp2_face_regression_lab_match_6h"
while screen -ls 2>/dev/null | grep -q '[.]exp2_face_regression_lab_match_6h[[:space:]]'; do
    sleep 30
done
if [[ ! -f "${BASELINE_DIR}/COMPLETE" ||
      ! -f "${BASELINE_DIR}/figures/shared_test_comparison.png" ]]; then
    echo "[monitor-error] 6h baseline screen exited without complete results" >&2
    exit 1
fi
echo "[monitor-ready] 6h baseline complete; starting 12h schedule ablation"
"${PYTHON}" -u -m study.exp2_face_pretrained_head32_regression.run_match_window_patient_diverse_schedule --hours 12
echo "[monitor-next] starting 6h schedule ablation"
"${PYTHON}" -u -m study.exp2_face_pretrained_head32_regression.run_match_window_patient_diverse_schedule --hours 6
echo "[all-window-schedule-ablations-complete]"
