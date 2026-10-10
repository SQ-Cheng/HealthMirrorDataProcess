#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="exp6_full24h_after_exp2"
LOGS="$ROOT/study/exp6_face_pair_lab_delta/logs/face224"
cd "$ROOT"
if screen -list | grep -q "[.]$SESSION[[:space:]]"; then
  echo "Screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HEALTHMIRROR_FACE_SOURCE=face224
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m \
    study.exp6_face_pair_lab_delta.run_after_current \
    2>&1 | tee -a "$1/autostart.log"
  status=$?
  printf "[monitor-exit] status=%s\n" "$status" | tee -a "$1/autostart.log"
  exit "$status"
' _ "$LOGS"
printf 'Detached screen: %s\nMonitor log: %s/autostart.log\nTraining log: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOGS" "$LOGS" "$SESSION"
