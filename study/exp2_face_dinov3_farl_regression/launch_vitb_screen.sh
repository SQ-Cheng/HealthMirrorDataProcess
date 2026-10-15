#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_dinov3b_farl_regression"
LOGS="$ROOT/study/exp2_face_dinov3_farl_regression/logs/vitb16"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -euo pipefail
  cd "$1"
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python=/root/miniconda3/envs/healthmirrorenv/bin/python
  exec > >(tee -a "$2/run.log") 2>&1
  printf "[queue-start] %s frozen ViT-B + FaRL: concat/vector/scalar, all10 targets\n" "$(date -Is)"
  "$python" -u -m study.exp2_face_dinov3_farl_regression.run --backbone dinov3_vitb16 --smoke
  "$python" -u -m study.exp2_face_dinov3_farl_regression.run --backbone dinov3_vitb16
  printf "[queue-complete] %s thirty heads and updated comparison figures completed\n" "$(date -Is)"
' _ "$ROOT" "$LOGS"
echo "Detached screen: $SESSION"
echo "Log: $LOGS/run.log"
