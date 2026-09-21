#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp5_face_pair_recovery"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

cd "$ROOT"
mkdir -p "$EXP/logs/ablations"
"$PYTHON" -m study.exp5_face_pair_recovery.run_all --prepare-only

"$PYTHON" -m study.exp5_face_pair_recovery.run_all --device 0 \
  > >(tee "$EXP/logs/paired.log") 2>&1 &
paired_pid=$!
"$PYTHON" -m study.exp5_face_pair_recovery.ablation_train \
  --mode pre_only --device 1 > >(tee "$EXP/logs/ablations/pre_only.log") 2>&1 &
pre_pid=$!
"$PYTHON" -m study.exp5_face_pair_recovery.ablation_train \
  --mode post_only --device 2 > >(tee "$EXP/logs/ablations/post_only.log") 2>&1 &
post_pid=$!

status=0
wait "$paired_pid" || status=1
wait "$pre_pid" || status=1
wait "$post_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "[training-failed] at least one protocol failed" >&2
  exit 1
fi

"$PYTHON" -m study.exp5_face_pair_recovery.plot_ablation_comparison
echo "[post-training-complete] three protocols and all figures completed"
