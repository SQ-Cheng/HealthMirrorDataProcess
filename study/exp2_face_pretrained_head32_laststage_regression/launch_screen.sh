#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
VARIANT="${1:-20frame}"
if (( $# > 0 )); then
    shift
fi
case "${VARIANT}" in
    20frame|allframes) ;;
    *)
        echo "usage: $0 {20frame|allframes} [run_all options]" >&2
        exit 2
        ;;
esac

SESSION_NAME="${SESSION_NAME:-exp2_face_pretrained_head32_laststage_regression_${VARIANT}}"
LOG_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_laststage_regression/logs/${VARIANT}"
LOG_FILE="${LOG_DIR}/run.log"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
EXTRA_ARGS=" --frame-policy ${VARIANT}"
if [[ "${VARIANT}" == "20frame" ]]; then
    REFERENCE_ROOT="${ROOT_DIR}/study/exp2_face_pretrained_head64_regression/outputs/20frame"
    EXTRA_ARGS+=" --source-dir ${REFERENCE_ROOT}/source_data"
    EXTRA_ARGS+=" --index-dir ${REFERENCE_ROOT}/frame_index"
    EXTRA_ARGS+=" --reference-records-dir ${REFERENCE_ROOT}/task_records"
fi
if (( $# > 0 )); then
    printf -v USER_ARGS ' %q' "$@"
    EXTRA_ARGS+="${USER_ARGS}"
fi
COMMAND="set -o pipefail; cd ${ROOT_DIR} && source /root/miniconda3/etc/profile.d/conda.sh && conda activate healthmirrorenv && CUDA_VISIBLE_DEVICES=0,1,2,3 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 python -u -m study.exp2_face_pretrained_head32_laststage_regression.run_all${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}"
screen -dmS "${SESSION_NAME}" bash -lc "${COMMAND}"
echo "Started detached screen session: ${SESSION_NAME}"
echo "Attach: screen -r ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
