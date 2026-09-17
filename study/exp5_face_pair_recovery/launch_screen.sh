#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp5_face_pair_recovery"
SESSION="exp5_face_pair_recovery"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$EXP/logs"
screen -dmS "$SESSION" bash -lc \
  "cd '$ROOT' && set -o pipefail && \
   export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 && \
   { '$PYTHON' -m study.exp5_face_pair_recovery.run_all && \
     '$PYTHON' -m study.exp5_face_pair_recovery.interpretability --examples 10 --device 0 && \
     '$PYTHON' -m study.exp5_face_pair_recovery.plot_ablation_comparison && \
     echo '[post-training-complete] all current figures regenerated'; \
   } 2>&1 | tee '$EXP/logs/run.log'"
echo "started detached screen: $SESSION"
echo "attach with: screen -r $SESSION"
