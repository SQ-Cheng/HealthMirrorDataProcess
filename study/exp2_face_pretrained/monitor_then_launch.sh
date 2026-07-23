#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
CURRENT_SESSION="${CURRENT_SESSION:-exp2_face_only_aug20_24h_native_hgb}"
NEXT_SESSION="${NEXT_SESSION:-exp2_face_pretrained}"
POLL_SECONDS="${POLL_SECONDS:-60}"

CURRENT_LOG="${ROOT_DIR}/study/exp2_face_only/logs_aug20_24h/run.log"
CURRENT_OUTPUT="${ROOT_DIR}/study/exp2_face_only/outputs_aug20_24h_views5_seed5"
NEXT_LAUNCHER="${ROOT_DIR}/study/exp2_face_pretrained/launch_screen.sh"
NEXT_WEIGHTS="${ROOT_DIR}/study/exp2_face_pretrained/pretrained_weights/manifest.json"
MONITOR_DIR="${ROOT_DIR}/study/exp2_face_pretrained/logs"
MONITOR_LOG="${MONITOR_DIR}/handoff_monitor.log"
LOCK_FILE="${MONITOR_DIR}/handoff_monitor.lock"

mkdir -p "${MONITOR_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another handoff monitor is already running." >&2
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
    [[ -f "${CURRENT_OUTPUT}/job_index.csv" ]] || return 1
    grep -q '^Saved two-seed results to ' "${CURRENT_LOG}" || return 1
    grep -q '^Total elapsed minutes: ' "${CURRENT_LOG}" || return 1
    awk -F, '
        NR > 1 && $1 ~ /^(20260702|20260703)$/ &&
        $2 == "hemoglobin_low" && $4 == "ok" { successful += 1 }
        END { exit !(successful == 2) }
    ' "${CURRENT_OUTPUT}/job_index.csv"
}

log "Monitoring screen=${CURRENT_SESSION}; poll_seconds=${POLL_SECONDS}"
while screen_exists "${CURRENT_SESSION}"; do
    last_epoch="$(grep '^\[epoch\]' "${CURRENT_LOG}" 2>/dev/null | tail -n 1 || true)"
    if [[ -n "${last_epoch}" ]]; then
        log "Current training active: ${last_epoch}"
    else
        log "Current training active; waiting for epoch output"
    fi
    sleep "${POLL_SECONDS}"
done

log "Current training screen has exited; validating completion artifacts"
if ! current_run_succeeded; then
    log "ERROR: current training did not pass completion checks; pretrained experiment will not start"
    exit 2
fi

if [[ ! -x "${NEXT_LAUNCHER}" || ! -f "${NEXT_WEIGHTS}" ]]; then
    log "ERROR: pretrained launcher or verified weight manifest is missing"
    exit 3
fi
if screen_exists "${NEXT_SESSION}"; then
    log "ERROR: next screen already exists: ${NEXT_SESSION}"
    exit 4
fi

log "Current training completed successfully; launching ${NEXT_SESSION}"
SESSION_NAME="${NEXT_SESSION}" bash "${NEXT_LAUNCHER}"
sleep 3
if ! screen_exists "${NEXT_SESSION}"; then
    log "ERROR: launcher returned but ${NEXT_SESSION} is not running"
    exit 5
fi
log "Handoff complete; pretrained experiment is running in ${NEXT_SESSION}"
