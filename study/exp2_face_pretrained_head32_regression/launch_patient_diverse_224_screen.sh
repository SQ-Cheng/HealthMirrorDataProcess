#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="face224_patient_diverse_30_40"
LOGS="$ROOT/study/exp2_face_pretrained_head32_regression/logs/ablations/patient_diverse_schedule_30_40_face224"
cd "$ROOT"
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -o pipefail
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HEALTHMIRROR_FACE_SOURCE=face224
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /root/miniconda3/envs/healthmirrorenv/bin/python -u -m \
    study.exp2_face_pretrained_head32_regression.run_patient_diverse_224 \
    2>&1 | tee -a "$1/run.log"
  status=${PIPESTATUS[0]}
  printf "[monitor-exit] status=%s\n" "$status" | tee -a "$1/run.log"
  exit "$status"
' _ "$LOGS"
printf 'Detached screen: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOGS" "$SESSION"
