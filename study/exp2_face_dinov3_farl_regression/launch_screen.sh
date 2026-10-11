#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_dinov3_farl_regression"
VARIANT="${1:-all}"
if [[ "$VARIANT" != all && "$VARIANT" != scalar_gated ]]; then echo "Usage: $0 [scalar_gated]" >&2; exit 1; fi
LOG_NAME="run.log"
if [[ "$VARIANT" == scalar_gated ]]; then
  SESSION="exp2_dinov3_farl_scalar_gate"
  LOG_NAME="scalar_gate_run.log"
fi
LOGS="$ROOT/study/exp2_face_dinov3_farl_regression/logs"
cd "$ROOT"
export TERM=xterm
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then echo "Screen already exists" >&2; exit 1; fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python=/root/miniconda3/envs/healthmirrorenv/bin/python
  args=()
  if [[ "$2" == scalar_gated ]]; then args=(--variants scalar_gated); fi
  printf "[run-start] %s variant=%s\n" "$(date -Is)" "$2" | tee -a "$1/$3"
  "$python" -u -m study.exp2_face_dinov3_farl_regression.run --smoke "${args[@]}" 2>&1 | tee -a "$1/$3"
  status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    "$python" -u -m study.exp2_face_dinov3_farl_regression.run "${args[@]}" 2>&1 | tee -a "$1/$3"
    status=${PIPESTATUS[0]}
  fi
  printf "[queue-exit] status=%s\n" "$status" | tee -a "$1/$3"
  exit "$status"
' _ "$LOGS" "$VARIANT" "$LOG_NAME"
sleep 1
printf 'Detached screen: %s\nLog: %s/%s\n' "$SESSION" "$LOGS" "$LOG_NAME"
