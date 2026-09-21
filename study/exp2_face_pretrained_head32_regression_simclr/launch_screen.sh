#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp2_face_pretrained_head32_regression_simclr"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"
SESSION="exp2_head32_regression_simclr"

if screen -list 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$EXP/logs"
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && set -o pipefail && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU \
          OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
          PYTHONUNBUFFERED=1 && \
   '$PYTHON' -u -m study.exp2_face_pretrained_head32_regression_simclr.run_all \
     --gpus 0,1,2,3 --overwrite 2>&1 | tee '$EXP/logs/run.log'"
echo "started detached screen: $SESSION"
echo "attach with: screen -r $SESSION"
echo "log: $EXP/logs/run.log"
