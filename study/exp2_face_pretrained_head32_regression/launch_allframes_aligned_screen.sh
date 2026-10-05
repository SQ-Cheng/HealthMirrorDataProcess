#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
SESSION_NAME="exp2_face_regression_allframes_aligned"
LOG_ROOT="${EXP_DIR}/logs/allframes_aligned"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    exit 1
fi
mkdir -p "${LOG_ROOT}"
screen -dmS "${SESSION_NAME}" bash -lc \
    "set -o pipefail; '${EXP_DIR}/run_allframes_aligned_queue.sh' 2>&1 | tee '${LOG_ROOT}/queue.log'"
echo "Started detached screen: ${SESSION_NAME}"
echo "Queue log: ${LOG_ROOT}/queue.log"
