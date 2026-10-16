#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_rgb_ir_dinov3b"
LOGS="$ROOT/study/exp2_face_rgb_ir_dinov3b_regression/logs"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -euo pipefail
  cd "$1"
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  exec > >(tee -a "$2/run.log") 2>&1
  trap '\''status=$?; printf "[queue-exit] %s status=%s\n" "$(date -Is)" "$status"'\'' EXIT
  printf "[queue-start] %s independent frozen RGB/IR DINOv3-B (+FaRL); 6 structures x10 tasks; 8 GPUs\n" "$(date -Is)"
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp2_face_rgb_ir_dinov3b_regression.run --workers 8 --smoke
' _ "$ROOT" "$LOGS"
echo "Detached screen: $SESSION"
echo "Log: $LOGS/run.log"
