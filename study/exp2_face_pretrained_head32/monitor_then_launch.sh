#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
CURRENT_SESSION="${CURRENT_SESSION:-exp2_face_pretrained}"
NEXT_SESSION="${NEXT_SESSION:-exp2_face_pretrained_head32}"
POLL_SECONDS="${POLL_SECONDS:-60}"

CURRENT_LOG="${ROOT_DIR}/study/exp2_face_pretrained/logs/run.log"
CURRENT_OUTPUT="${ROOT_DIR}/study/exp2_face_pretrained/outputs"
NEXT_LAUNCHER="${ROOT_DIR}/study/exp2_face_pretrained_head32/launch_screen.sh"
NEXT_OUTPUT="${ROOT_DIR}/study/exp2_face_pretrained_head32/outputs"
MONITOR_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32/logs"
MONITOR_LOG="${MONITOR_DIR}/handoff_monitor.log"
LOCK_FILE="${MONITOR_DIR}/handoff_monitor.lock"

mkdir -p "${MONITOR_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another head-32 handoff monitor is already running." >&2
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${MONITOR_LOG}"
}

screen_exists() {
    screen -ls 2>/dev/null | grep -q "[.]${1}[[:space:]]"
}

current_run_succeeded() {
    [[ -f "${CURRENT_LOG}" ]] || return 1
    grep -q '^Experiment outputs saved to ' "${CURRENT_LOG}" || return 1
    python3 - "${CURRENT_OUTPUT}" <<'PY'
import csv
import json
import os
import sys

output_dir = sys.argv[1]
with open(os.path.join(output_dir, "experiment_manifest.json"), encoding="utf-8") as handle:
    manifest = json.load(handle)
expected = {
    (architecture, target)
    for architecture in manifest["architectures"]
    for target in manifest["ready_targets"]
}
with open(os.path.join(output_dir, "run_index.csv"), newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
actual = {(row["architecture"], row["target"]) for row in rows}
if actual != expected or any(row["status"] != "ok" for row in rows):
    raise SystemExit(1)
for row in rows:
    for filename in ("model.pt", "history.csv", "history.png", "metrics.csv"):
        if not os.path.isfile(os.path.join(row["run_dir"], filename)):
            raise SystemExit(1)
print(f"validated {len(rows)} successful jobs")
PY
}

log "Monitoring screen=${CURRENT_SESSION}; poll_seconds=${POLL_SECONDS}"
while screen_exists "${CURRENT_SESSION}"; do
    progress="$(grep '^\[scheduler\]' "${CURRENT_LOG}" 2>/dev/null | tail -n 1 || true)"
    if [[ -n "${progress}" ]]; then
        log "Current experiment active: ${progress}"
    else
        log "Current experiment active; waiting for completed jobs"
    fi
    sleep "${POLL_SECONDS}"
done

log "Current experiment screen has exited; validating all completion artifacts"
if ! current_run_succeeded >>"${MONITOR_LOG}" 2>&1; then
    log "ERROR: current experiment is incomplete or failed; head-32 experiment will not start"
    exit 2
fi
if [[ -f "${NEXT_OUTPUT}/run_index.csv" ]]; then
    log "ERROR: head-32 output already contains a run; refusing to overwrite it"
    exit 3
fi
if [[ ! -x "${NEXT_LAUNCHER}" ]]; then
    log "ERROR: head-32 launcher is missing or not executable"
    exit 4
fi
if screen_exists "${NEXT_SESSION}"; then
    log "ERROR: next screen already exists: ${NEXT_SESSION}"
    exit 5
fi

log "Current experiment completed successfully; launching ${NEXT_SESSION}"
SESSION_NAME="${NEXT_SESSION}" bash "${NEXT_LAUNCHER}"
sleep 3
if ! screen_exists "${NEXT_SESSION}"; then
    log "ERROR: launcher returned but ${NEXT_SESSION} is not running"
    exit 6
fi
log "Handoff complete; head-32 experiment is running in ${NEXT_SESSION}"
