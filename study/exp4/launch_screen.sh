#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
EXP_DIR="$ROOT_DIR/study/exp4"
SESSION="exp4_recovery"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$EXP_DIR/logs"
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT_DIR' && set -o pipefail && \
   export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 && \
   '$PYTHON' -m study.exp4.run_all 2>&1 | tee '$EXP_DIR/logs/run.log'"
echo "started screen session: $SESSION"
echo "attach with: screen -r $SESSION"
