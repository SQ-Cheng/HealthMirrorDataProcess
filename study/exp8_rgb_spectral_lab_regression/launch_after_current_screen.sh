#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp8_face224_12h_autostart"
LOGS="$ROOT/study/exp8_rgb_spectral_lab_regression/logs_autostart"
cd "$ROOT"
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp8_rgb_spectral_lab_regression.run_after_current \
    2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[monitor-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
