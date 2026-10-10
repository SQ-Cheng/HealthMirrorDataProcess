#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp1_roi_color_regression"
LOGS="$ROOT/study/exp1_roi_color_regression/logs"
cd "$ROOT"
export TERM=xterm
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then echo "Screen already exists" >&2; exit 1; fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3
  printf "[run-start] %s Exp1 native41 ROI regression\n" "$(date -Is)" | tee -a "$1/run.log"
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp1_roi_color_regression.run 2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
