#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp6_dinov3_regression"
LOGS="$ROOT/study/exp6_face_pair_dinov3_regression/logs"
cd "$ROOT"
export TERM="${TERM:-xterm}"
if [[ "$TERM" == dumb ]]; then export TERM=xterm; fi
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3
  unset HEALTHMIRROR_LAB_RUN_TAG HEALTHMIRROR_LAB_OVERWRITE
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python=/root/miniconda3/envs/healthmirrorenv/bin/python
  printf "[run-start] %s Exp6 frozen DINO head32/head64\n" "$(date -Is)" | tee -a "$1/run.log"
  "$python" -u -m study.exp6_face_pair_dinov3_regression.run --smoke 2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    "$python" -u -m study.exp6_face_pair_dinov3_regression.run 2>&1 | tee -a "$1/run.log"
    status=${PIPESTATUS[0]}
  fi
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
