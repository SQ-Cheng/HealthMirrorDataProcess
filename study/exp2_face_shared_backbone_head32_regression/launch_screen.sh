#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="exp2_shared_backbone_24h"
LOGS="$ROOT/study/exp2_face_shared_backbone_head32_regression/logs"
cd "$ROOT"
export TERM=xterm
if screen -list | grep -q "[.]$SESSION[[:space:]]"; then
  echo "Screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  log_dir="$1"
  shift
  overwrite=false
  for argument in "$@"; do if [[ "$argument" == --overwrite ]]; then overwrite=true; fi; done
  if $overwrite; then : > "$log_dir/run.log"; fi
  export CUDA_VISIBLE_DEVICES=0 HEALTHMIRROR_FACE_SOURCE=face224
  export MKL_THREADING_LAYER=GNU OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python=/root/miniconda3/envs/healthmirrorenv/bin/python
  printf "[run-start] %s overwrite=%s shared EN-B0 / ten masked regression heads\n" "$(date -Is)" "$overwrite" | tee -a "$log_dir/run.log"
  status=0
  if $overwrite; then
    "$python" -u -m study.exp2_face_shared_backbone_head32_regression.run_all --overwrite --prepare-only 2>&1 | tee -a "$log_dir/run.log"
    status=$?
    if [[ "$status" -eq 0 ]]; then
      "$python" -u -m study.exp2_face_shared_backbone_head32_regression.run_all --reuse-prepared --compiled-smoke 2>&1 | tee -a "$log_dir/run.log"
      status=$?
    fi
    if [[ "$status" -eq 0 ]]; then
      "$python" -u -m study.exp2_face_shared_backbone_head32_regression.run_all --reuse-prepared 2>&1 | tee -a "$log_dir/run.log"
      status=$?
    fi
  else
    "$python" -u -m study.exp2_face_shared_backbone_head32_regression.run_all --reuse-prepared "$@" 2>&1 | tee -a "$log_dir/run.log"
    status=$?
  fi
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$log_dir/run.log"
  exit "$status"
' _ "$LOGS" "$@"
printf 'Detached screen: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOGS" "$SESSION"
