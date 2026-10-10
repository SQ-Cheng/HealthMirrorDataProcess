#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_dinov3_main24h_regression"
LOGS="$ROOT/study/exp2_face_dinov3_frozen/logs/main24h_regression"
HEAD_HIDDEN="${1:-32}"
case "$HEAD_HIDDEN" in
  32) ;;
  64) SESSION="${SESSION}_head64"; LOGS="${LOGS}_head64" ;;
  *) echo "Head width must be 32 or 64" >&2; exit 1 ;;
esac
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
  printf "[run-start] %s DINO main24h frame-level regression head=%s\n" "$(date -Is)" "$2" | tee -a "$1/run.log"
  "$python" -u -m study.exp2_face_dinov3_frozen.run_main_regression --head-hidden "$2" --smoke 2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    "$python" -u -m study.exp2_face_dinov3_frozen.run_main_regression --head-hidden "$2" 2>&1 | tee -a "$1/run.log"
    status=${PIPESTATUS[0]}
  fi
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS" "$HEAD_HIDDEN"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
