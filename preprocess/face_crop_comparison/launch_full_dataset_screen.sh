#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="face224_full_c075"
LOG_DIR="$ROOT/preprocess/face_crop_comparison/logs/full_dataset"
mkdir -p "$LOG_DIR"
cd "$ROOT"
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  printf 'Screen session already exists: %s\n' "$SESSION" >&2
  exit 1
fi
screen -dmS "$SESSION" bash -c '
  LOG_DIR=$1
  shift
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m \
    preprocess.face_crop_comparison.process_full_dataset --workers 8 "$@" \
    2>&1 | tee -a "$LOG_DIR/run.log"
  status=${PIPESTATUS[0]}
  printf "[exit] status=%s\n" "$status" | tee -a "$LOG_DIR/run.log"
  exit "$status"
' _ "$LOG_DIR" "$@"
printf 'Started: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOG_DIR" "$SESSION"
