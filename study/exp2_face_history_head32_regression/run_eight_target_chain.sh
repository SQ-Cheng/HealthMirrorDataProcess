#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
TARGETS="oxyhemoglobin_fraction,lactate_high,urea_high,troponin_high,platelet_count_low,hemoglobin_low,aa_po2_ratio_low,creatinine_high"
CHAIN_LOG="${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/eight_target_chain.log"

mkdir -p "$(dirname "${CHAIN_LOG}")"
exec > >(tee "${CHAIN_LOG}") 2>&1

cd "${ROOT_DIR}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate healthmirrorenv
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1

run_stage() {
    local name="$1"
    local log_file="$2"
    shift 2
    mkdir -p "$(dirname "${log_file}")"
    printf '[chain-stage-start] %s stage=%s\n' "$(date -Is)" "${name}"
    "$@" 2>&1 | tee "${log_file}"
    printf '[chain-stage-complete] %s stage=%s\n' "$(date -Is)" "${name}"
}

run_stage \
    face_history \
    "${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/20frame/run.log" \
    python -u -m study.exp2_face_history_head32_regression.run_all \
        --architectures efficientnet_b0 \
        --targets "${TARGETS}" \
        --overwrite \
        --rebuild-split

run_stage \
    face_only \
    "${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/logs/20frame/run.log" \
    python -u -m study.exp2_face_pretrained_head32_regression.run_all \
        --frame-policy 20frame \
        --architectures efficientnet_b0 \
        --targets "${TARGETS}" \
        --overwrite

run_stage \
    history_only \
    "${ROOT_DIR}/study/exp2_history_only_head32_regression/logs/run.log" \
    python -u -m study.exp2_history_only_head32_regression.run_all \
        --targets "${TARGETS}" \
        --overwrite

printf '[chain-complete] %s\n' "$(date -Is)"
