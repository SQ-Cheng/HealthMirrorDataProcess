#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="face224_experiments"
LOGS="$ROOT/study/common/logs/face224_reruns"
cd "$ROOT"
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HEALTHMIRROR_FACE_SOURCE=face224
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.common.rerun_face224 \
    2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
printf 'Detached screen: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOGS" "$SESSION"
