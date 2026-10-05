#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp6_face_pair_lab_delta"
SESSION="exp6_lab_match_12h_6h"
LOG_DIR="${EXP_DIR}/logs/ablations/match_windows"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "screen session already exists: ${SESSION}" >&2
    exit 1
fi
for hours in 12 6; do
    output="${EXP_DIR}/outputs/ablations/lab_match_${hours}h"
    if [[ ! -f "${output}/PREPARED" || -e "${output}/runs" ]]; then
        echo "Expected prepared, untrained ${hours}h output: ${output}" >&2
        exit 1
    fi
done
mkdir -p "${LOG_DIR}"
screen -dmS "${SESSION}" bash -lc \
    "set -o pipefail; cd '${ROOT_DIR}'; export CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1; ( '${PYTHON}' -u -m study.exp6_face_pair_lab_delta.run_match_window_ablation --hours 12 --train-only && '${PYTHON}' -u -m study.exp6_face_pair_lab_delta.run_match_window_ablation --hours 6 --train-only ) 2>&1 | tee '${LOG_DIR}/run.log'"
echo "Started detached screen: ${SESSION}"
echo "Log: ${LOG_DIR}/run.log"
