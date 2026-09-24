#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"
TARGETS="oxyhemoglobin_fraction,lactate_high,urea_high,total_bilirubin_high,platelet_count_low,hemoglobin_low,aa_po2_ratio_low,creatinine_high"
CHAIN_LOG="${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/time_alignment_fixed_chain.log"

mkdir -p "$(dirname "${CHAIN_LOG}")"
exec > >(tee "${CHAIN_LOG}") 2>&1

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
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
    face_history_regression \
    "${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/20frame/run.log" \
    "${PYTHON}" -u -m study.exp2_face_history_head32_regression.run_all \
        --frame-policy 20frame \
        --architectures efficientnet_b0 \
        --targets "${TARGETS}" \
        --overwrite \
        --rebuild-split

run_stage \
    face_only_regression \
    "${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/logs/20frame/run.log" \
    "${PYTHON}" -u -m study.exp2_face_pretrained_head32_regression.run_all \
        --frame-policy 20frame \
        --architectures efficientnet_b0 \
        --targets "${TARGETS}" \
        --overwrite

run_stage \
    history_only_regression \
    "${ROOT_DIR}/study/exp2_history_only_head32_regression/logs/run.log" \
    "${PYTHON}" -u -m study.exp2_history_only_head32_regression.run_all \
        --targets "${TARGETS}" \
        --overwrite

run_stage \
    three_binary_ablations \
    "${ROOT_DIR}/study/exp2_face_history_head32_classification/logs/run.log" \
    "${PYTHON}" -u -m study.exp2_binary_classification_common.run_all \
        --gpus 0,1,2,3 \
        --overwrite

run_stage \
    exp6_shared_backbone \
    "${ROOT_DIR}/study/exp6_face_pair_lab_delta/logs/shared.log" \
    "${PYTHON}" -u -m study.exp6_face_pair_lab_delta.run_all \
        --variant shared \
        --overwrite

run_stage \
    exp6_independent_backbones \
    "${ROOT_DIR}/study/exp6_face_pair_lab_delta/logs/independent_backbones.log" \
    "${PYTHON}" -u -m study.exp6_face_pair_lab_delta.run_all \
        --variant independent_backbones \
        --overwrite

printf '[chain-complete] %s\n' "$(date -Is)"
