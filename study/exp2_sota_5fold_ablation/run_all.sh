#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export HEALTHMIRROR_FACE_SOURCE=face224
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export TORCHINDUCTOR_COMPILE_THREADS=1
exec /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp2_sota_5fold_ablation.run --run "$@"
