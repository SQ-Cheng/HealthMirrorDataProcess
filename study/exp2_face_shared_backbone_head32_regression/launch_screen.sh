#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="exp2_shared_backbone_24h"
LOGS="$ROOT/study/exp2_face_shared_backbone_head32_regression/logs"
cd "$ROOT"
if screen -list | grep -q "[.]$SESSION[[:space:]]"; then
  echo "Screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  log_dir="$1"
  shift
  export CUDA_VISIBLE_DEVICES=0 HEALTHMIRROR_FACE_SOURCE=face224
  export MKL_THREADING_LAYER=GNU OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m \
    study.exp2_face_shared_backbone_head32_regression.run_all --reuse-prepared "$@" \
    2>&1 | tee -a "$log_dir/run.log"
  status=$?
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$log_dir/run.log"
  exit "$status"
' _ "$LOGS" "$@"
printf 'Detached screen: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOGS" "$SESSION"
