#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${HEALTHMIRROR_LAB_RUN_TAG:-lab_update_20261007}"
SESSION="exp2_exp4_${TAG}"
LOGS="$ROOT/study/common/logs/$TAG"
cd "$ROOT"
export TERM="${TERM:-xterm}"
if [[ "$TERM" == "dumb" ]]; then export TERM=xterm; fi
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 HEALTHMIRROR_LAB_RUN_TAG="$2" HEALTHMIRROR_LAB_OVERWRITE=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.common.run_latest_lab_queue \
    2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS" "$TAG"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
