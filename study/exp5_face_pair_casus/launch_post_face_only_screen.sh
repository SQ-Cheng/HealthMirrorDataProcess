#!/usr/bin/env bash
set -euo pipefail

root="/root/autodl-tmp/HealthMirrorDataProcess"
python="/root/miniconda3/envs/healthmirrorenv/bin/python"
session="exp5_casus_post_face_only"
log="study/exp5_face_pair_casus/logs/post_face_only_run.log"
cd "$root"
mkdir -p "$(dirname "$log")"
if screen -list | grep -q "[.]${session}[[:space:]]"; then
  echo "screen session already exists: ${session}" >&2
  exit 1
fi
: > "$log"
screen -dmS "$session" bash -lc \
  "cd '$root' && set -o pipefail && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; '$python' -m study.exp5_face_pair_casus.run_post_face_only 2>&1 | tee '$log'"
echo "started screen session: ${session}"
echo "attach with: screen -r ${session}"
