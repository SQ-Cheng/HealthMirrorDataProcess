#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp2_face_pretrained_head32_regression_simclr"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"
WATCH_SESSION="exp2_three_binary"

mkdir -p "$EXP/logs"
echo "[monitor] waiting for screen=$WATCH_SESSION at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
while screen -list 2>/dev/null | grep -q "[.]${WATCH_SESSION}[[:space:]]"; do
  sleep 60
done
echo "[monitor] binary experiment ended at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
sleep 15

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[monitor] starting bounded smoke test"
set -o pipefail
"$PYTHON" -u -m study.exp2_face_pretrained_head32_regression_simclr.run_all \
  --smoke-test --gpus 0 --overwrite 2>&1 | tee "$EXP/logs/smoke.log"

echo "[monitor] smoke test passed; starting formal four-GPU run"
"$PYTHON" -u -m study.exp2_face_pretrained_head32_regression_simclr.run_all \
  --gpus 0,1,2,3 --overwrite 2>&1 | tee "$EXP/logs/run.log"
echo "[monitor] formal run complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
