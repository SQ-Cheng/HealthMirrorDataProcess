#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_face_main_24h_frame_loss"
LOGS="$ROOT/study/common/logs/face_main_24h_frame_loss"
cd "$ROOT"
export TERM="${TERM:-xterm}"
if [[ "$TERM" == "dumb" ]]; then export TERM=xterm; fi
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
if [[ $# == 0 ]]; then set -- --overwrite; fi
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  unset HEALTHMIRROR_LAB_RUN_TAG HEALTHMIRROR_LAB_OVERWRITE
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.common.run_face_main_24h "${@:2}" \
    2>&1 | tee "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS" "$@"
sleep 1
printf 'Detached screen: %s\nLog: %s/run.log\n' "$SESSION" "$LOGS"
