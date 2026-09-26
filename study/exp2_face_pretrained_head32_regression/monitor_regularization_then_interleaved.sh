#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ABLATION_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/outputs/ablations"
CURRENT_SESSION="exp2_face_regression_regularization"

while screen -ls 2>/dev/null | grep -q "[.]${CURRENT_SESSION}[[:space:]]"; do
    sleep 30
done

for variant in patient_diverse_schedule_30_40_wd1e2 patient_diverse_schedule_30_40_bn_fixed; do
    if [[ ! -f "${ABLATION_DIR}/${variant}/COMPLETE" ]]; then
        echo "[not-started] Required experiment did not finish: ${variant}" >&2
        exit 1
    fi
done

echo "[launch] Both regularization ablations complete; starting interleaved-view experiment"
cd "${ROOT_DIR}"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 \
    /root/miniconda3/envs/healthmirrorenv/bin/python -u \
    -m study.exp2_face_pretrained_head32_regression.run_interleaved_views_ablation
