#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
EXP="$ROOT/study/exp5_face_pair_recovery"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

mkdir -p "$EXP/logs/ablations" "$EXP/outputs/ablations/post_only" "$EXP/outputs/ablations/pre_only"

"$PYTHON" -m study.exp5_face_pair_recovery.ablation_train \
  --mode post_only --device 0 2>&1 | tee "$EXP/logs/ablations/post_only.log" &
post_pid=$!
"$PYTHON" -m study.exp5_face_pair_recovery.ablation_train \
  --mode pre_only --device 1 2>&1 | tee "$EXP/logs/ablations/pre_only.log" &
pre_pid=$!

wait "$post_pid"
wait "$pre_pid"
"$PYTHON" -m study.exp5_face_pair_recovery.plot_ablation_comparison
echo "[ablations-complete] both runs and comparison figure generated"
