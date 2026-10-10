#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_dinov3_farl_regression"
LOGS="$ROOT/study/exp2_face_dinov3_farl_regression/logs"
cd "$ROOT"
export TERM=xterm
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then echo "Screen already exists" >&2; exit 1; fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python=/root/miniconda3/envs/healthmirrorenv/bin/python
  "$python" -u -m study.exp2_face_dinov3_farl_regression.run --smoke 2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    "$python" -u -m study.exp2_face_dinov3_farl_regression.run 2>&1 | tee -a "$1/run.log"
    status=${PIPESTATUS[0]}
  fi
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
