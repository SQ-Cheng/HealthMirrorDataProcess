#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
EXP="${ROOT}/study/exp8_rgb_spectral_lab_regression"
VARIANT="${1:-ntire2022}"
GRID_SIZE="${2:-4}"
if [[ "${VARIANT}" != "ntire2022" && "${VARIANT}" != "hyperskin" ]]; then
    echo "Usage: $0 [ntire2022|hyperskin] [4|16]" >&2
    exit 2
fi
if [[ "${GRID_SIZE}" != "4" && "${GRID_SIZE}" != "16" ]]; then
    echo "Usage: $0 [ntire2022|hyperskin] [4|16]" >&2
    exit 2
fi
SESSION="exp8_rgb_spectral_lab_${VARIANT}"
SUFFIX=""
SOURCE_SUFFIX="$('/root/miniconda3/envs/healthmirrorenv/bin/python' -c 'from study.common.face_video import face_source_mode; print("_face224" if face_source_mode() == "face224" else "")')"
if [[ "${VARIANT}" == "hyperskin" ]]; then
    SUFFIX="_hyperskin"
    WEIGHT="${ROOT}/study/common/pretrained_weights/mst_plus_plus_hyperskin_rgb_vis.pth"
    if [[ ! -s "${WEIGHT}" ]]; then
        echo "Missing Hyper-Skin RGB-to-VIS MST++ checkpoint: ${WEIGHT}" >&2
        exit 1
    fi
fi
if [[ "${GRID_SIZE}" != "4" ]]; then
    SUFFIX="${SUFFIX}_grid${GRID_SIZE}"
    SESSION="${SESSION}_grid${GRID_SIZE}"
fi
SUFFIX="${SUFFIX}${SOURCE_SUFFIX}"
SESSION="${SESSION}${SOURCE_SUFFIX}"
OUTPUT="${EXP}/outputs${SUFFIX}"
LOGS="${EXP}/logs${SUFFIX}"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen session already exists: ${SESSION}" >&2
    exit 1
fi
if [[ -e "${OUTPUT}/COMPLETE" ]]; then
    echo "Exp8 output is already complete; refusing to overwrite it" >&2
    exit 1
fi
mkdir -p "${LOGS}"
screen -dmS "${SESSION}" bash -lc \
    "set -o pipefail; cd '${ROOT}' && EXP8_VARIANT='${VARIANT}' EXP8_GRID_SIZE='${GRID_SIZE}' PYTHONUNBUFFERED=1 /root/miniconda3/envs/healthmirrorenv/bin/python -u -m study.exp8_rgb_spectral_lab_regression.train 2>&1 | tee '${LOGS}/run.log'"
echo "Started detached screen: ${SESSION}"
echo "Log: ${LOGS}/run.log"
