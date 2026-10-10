#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="frozen_en_exp2_exp6_regression"
LOGS="$ROOT/study/common/logs/frozen_en_regression_controls"
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
  printf "[run-start] %s frozen EN-B0 Exp2/Exp6 regression controls\n" "$(date -Is)" | tee -a "$1/run.log"
  "$python" -u -m study.common.run_frozen_en_regression_controls --smoke 2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    "$python" -u -m study.common.run_frozen_en_regression_controls 2>&1 | tee -a "$1/run.log"
    status=${PIPESTATUS[0]}
  fi
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
