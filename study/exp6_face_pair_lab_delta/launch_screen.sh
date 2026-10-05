#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p study/exp6_face_pair_lab_delta/logs/face224
screen -dmS exp6_native224 bash -c 'set -o pipefail; export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU; /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp6_face_pair_lab_delta.run_all "$@" 2>&1 | tee -a study/exp6_face_pair_lab_delta/logs/face224/run.log' _ "$@"
