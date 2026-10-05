#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"
INDEX_DIR="${EXP_DIR}/cache/allframes_index"
LOG_ROOT="${EXP_DIR}/logs/allframes_aligned"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_ROOT}"

run_window() {
    local hours="$1"
    local reference="$2"
    local output="$3"
    local log_dir="${LOG_ROOT}/${hours}h"
    mkdir -p "${log_dir}"
    echo "[window-start] hours=${hours} reference=${reference} output=${output}"
    "${PYTHON}" -u -m study.exp2_face_pretrained_head32_regression.run_all \
        --frame-policy allframes \
        --match-max-delta-hours "${hours}" \
        --source-dir "${reference}/source_data" \
        --reuse-source-data \
        --reference-output-dir "${reference}" \
        --match-reference-samples \
        --index-dir "${INDEX_DIR}" \
        --output-dir "${output}" \
        --overwrite \
        --workers 4 \
        2>&1 | tee "${log_dir}/run.log"
    "${PYTHON}" -u -m study.exp2_face_pretrained_head32_regression.plot_allframes_comparison \
        --reference-dir "${reference}" \
        --candidate-dir "${output}" \
        --hours "${hours}" \
        2>&1 | tee "${log_dir}/comparison.log"
    touch "${log_dir}/COMPLETE"
    echo "[window-complete] hours=${hours}"
}

run_window 24 "${EXP_DIR}/outputs/20frame" "${EXP_DIR}/outputs/allframes"
run_window 12 "${EXP_DIR}/outputs/ablations/lab_match_12h" \
    "${EXP_DIR}/outputs/ablations/lab_match_12h_allframes"
run_window 6 "${EXP_DIR}/outputs/ablations/lab_match_6h" \
    "${EXP_DIR}/outputs/ablations/lab_match_6h_allframes"
touch "${LOG_ROOT}/COMPLETE"
echo "[allframes-aligned-complete]"
