#!/usr/bin/env bash
set -euo pipefail

root="/root/autodl-tmp/HealthMirrorDataProcess"
python="/root/miniconda3/envs/healthmirrorenv/bin/python"
cd "$root"
session="exp5_face_pair_casus"
mkdir -p study/exp5_face_pair_casus/logs
if screen -list | grep -q "[.]${session}[[:space:]]"; then
  echo "screen session already exists: ${session}" >&2
  exit 1
fi
: > study/exp5_face_pair_casus/logs/run.log
screen -dmS "$session" bash -lc \
  "cd '$root' && set -o pipefail && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; '$python' -m study.exp5_face_pair_casus.run_all 2>&1 | tee study/exp5_face_pair_casus/logs/run.log"
echo "started screen session: ${session}"
echo "attach with: screen -r ${session}"
