#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp1_dino_roi_views5"
cd "$ROOT"
export TERM=xterm
for name in "$SESSION" exp1_roi_color_regression; do
  if screen -list | grep -q "[.]${name}[[:space:]]"; then echo "Screen already exists: $name" >&2; exit 1; fi
done
mkdir -p study/exp1_roi_color_regression/logs study/exp2_face_dinov3_roi_color_regression/logs
screen -dmS "$SESSION" bash -c '
  set -euo pipefail
  export HEALTHMIRROR_FACE_SOURCE=face224 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3
  PY=/root/miniconda3/envs/healthmirrorenv/bin/python
  ROI=study/exp1_roi_color_regression/logs/run.log
  FUSION=study/exp2_face_dinov3_roi_color_regression/logs/run.log
  printf "[run-start] %s five-view ROI-only replacement and new DINO+ROI\n" "$(date -Is)" | tee "$ROI" "$FUSION"
  "$PY" -u -m study.exp1_roi_color_regression.run --prepare-only 2>&1 | tee -a "$ROI" "$FUSION"
  "$PY" -u -m study.exp2_face_dinov3_roi_color_regression.run --smoke 2>&1 | tee -a "$FUSION"
  # Both queues fit small cached-feature heads, not image encoders.
  "$PY" -u -m study.exp1_roi_color_regression.run --reuse-prepared 2>&1 | tee -a "$ROI" &
  roi_pid=$!
  "$PY" -u -m study.exp2_face_dinov3_roi_color_regression.run 2>&1 | tee -a "$FUSION" &
  fusion_pid=$!
  status=0
  wait "$roi_pid" || status=1
  wait "$fusion_pid" || status=1
  printf "[queue-exit] %s status=%s\n" "$(date -Is)" "$status" | tee -a "$ROI" "$FUSION"
  exit "$status"
'
sleep 1
printf 'Detached screen: %s\nROI log: study/exp1_roi_color_regression/logs/run.log\nFusion log: study/exp2_face_dinov3_roi_color_regression/logs/run.log\n' "$SESSION"
