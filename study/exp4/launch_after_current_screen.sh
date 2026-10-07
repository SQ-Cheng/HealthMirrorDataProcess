#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp4_recovery_224"
LOGS="$ROOT/study/exp4/logs"
cd "$ROOT"
if [[ "${TERM:-dumb}" == "dumb" ]]; then
  export TERM=xterm
fi
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp4.run_after_current \
    2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
