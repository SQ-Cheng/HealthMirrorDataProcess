#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
UPSTREAM_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
HISTORY_DIR="${ROOT_DIR}/study/exp2_face_history_head32_regression"
UPSTREAM_OUTPUT="${UPSTREAM_DIR}/outputs/20frame"
UPSTREAM_INDEX="${UPSTREAM_DIR}/cache/20frame_index"
UPSTREAM_TRAIN_SESSION="exp2_face_pretrained_head32_regression_20frame"
UPSTREAM_MONITOR_SESSION="exp2_face_pretrained_head32_regression_add_oxy_monitor"
HISTORY_SESSION="exp2_face_history_head32_regression_20frame"
LOG_FILE="${HISTORY_DIR}/logs/monitor_pretrained_then_add_oxyhemoglobin.log"

mkdir -p "$(dirname "${LOG_FILE}")"
printf '[%s] waiting for complete no-history 3-target run\n' "$(date -Is)" >> "${LOG_FILE}"

is_screen_running() {
    screen -ls 2>/dev/null | grep -q "[.]$1[[:space:]]"
}

upstream_complete() {
    python - "${UPSTREAM_OUTPUT}" <<'PY'
import os
import sys
import pandas as pd

output = sys.argv[1]
path = os.path.join(output, "run_index.csv")
if not os.path.isfile(path):
    raise SystemExit(1)
frame = pd.read_csv(path)
expected = {
    (architecture, target)
    for architecture in ("mobilenet_v3_small", "efficientnet_b0")
    for target in ("hemoglobin_low", "po2_low", "oxyhemoglobin_fraction")
}
successful = {
    (str(row.architecture), str(row.target))
    for row in frame.itertuples(index=False)
    if str(row.status) == "ok"
}
if successful != expected or len(frame) != len(expected):
    raise SystemExit(1)
for row in frame.itertuples(index=False):
    if not os.path.isfile(os.path.join(str(row.run_dir), "model.pt")):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

idle_polls=0
while ! upstream_complete; do
    if is_screen_running "${UPSTREAM_TRAIN_SESSION}" || is_screen_running "${UPSTREAM_MONITOR_SESSION}"; then
        idle_polls=0
    else
        idle_polls=$((idle_polls + 1))
        if (( idle_polls >= 5 )); then
            printf '[%s] ERROR: upstream sessions stopped before six successful jobs\n' "$(date -Is)" >> "${LOG_FILE}"
            exit 1
        fi
    fi
    sleep 30
done

printf '[%s] upstream complete; launching history oxyhemoglobin target\n' "$(date -Is)" >> "${LOG_FILE}"
if is_screen_running "${HISTORY_SESSION}"; then
    printf '[%s] ERROR: history training session already exists\n' "$(date -Is)" >> "${LOG_FILE}"
    exit 1
fi

SESSION_NAME="${HISTORY_SESSION}" bash "${HISTORY_DIR}/launch_screen.sh" \
    --targets oxyhemoglobin_fraction \
    --replace-targets \
    --reference-output-dir "${UPSTREAM_OUTPUT}" \
    --index-dir "${UPSTREAM_INDEX}" >> "${LOG_FILE}" 2>&1
printf '[%s] history launch submitted\n' "$(date -Is)" >> "${LOG_FILE}"
