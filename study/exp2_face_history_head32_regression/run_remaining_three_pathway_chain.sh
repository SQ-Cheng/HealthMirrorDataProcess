#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
TARGETS="oxyhemoglobin_fraction,lactate_high,urea_high,total_bilirubin_high,platelet_count_low,hemoglobin_low,aa_po2_ratio_low,creatinine_high"
CHAIN_LOG="${ROOT_DIR}/study/exp2_face_history_head32_regression/logs/remaining_three_pathway_chain.log"

mkdir -p "$(dirname "${CHAIN_LOG}")"
exec > >(tee "${CHAIN_LOG}") 2>&1

cd "${ROOT_DIR}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate healthmirrorenv
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1

printf '[chain-start] %s\n' "$(date -Is)"
python -u -m study.exp2_face_history_head32_regression.validate_and_plot_three_experiments \
    --stage face_history

printf '[stage-start] %s stage=face_only\n' "$(date -Is)"
python -u -m study.exp2_face_pretrained_head32_regression.run_all \
    --frame-policy 20frame \
    --architectures efficientnet_b0 \
    --targets "${TARGETS}" \
    --overwrite 2>&1 | tee \
    "${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/logs/20frame/run.log"
python -u -m study.exp2_face_history_head32_regression.validate_and_plot_three_experiments \
    --stage face_only
printf '[stage-complete] %s stage=face_only\n' "$(date -Is)"

printf '[stage-start] %s stage=history_only\n' "$(date -Is)"
python -u -m study.exp2_history_only_head32_regression.run_all \
    --targets "${TARGETS}" \
    --overwrite 2>&1 | tee \
    "${ROOT_DIR}/study/exp2_history_only_head32_regression/logs/run.log"
python -u -m study.exp2_face_history_head32_regression.validate_and_plot_three_experiments \
    --stage all \
    --generate-plots
printf '[stage-complete] %s stage=history_only\n' "$(date -Is)"
printf '[chain-complete] %s\n' "$(date -Is)"
