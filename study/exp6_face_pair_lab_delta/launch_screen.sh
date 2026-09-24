#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p study/exp6_face_pair_lab_delta/logs
exec /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp6_face_pair_lab_delta.run_all --overwrite
