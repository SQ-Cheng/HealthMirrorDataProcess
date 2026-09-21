#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/HealthMirrorDataProcess"
UPSTREAM="$ROOT/study/exp2_face_pretrained_head32_regression/outputs/20frame"
REFERENCE="$ROOT/study/exp2_face_history_head32_regression/outputs/20frame"
MONITOR_LOG="$ROOT/study/exp2_history_only_head32_regression/logs/monitor_pretrained_then_launch.log"
EXPECTED_LAB_SHA256="5e41a342586f85fe354ac668f23b1c3072790f560e2b6a2267bf71377dcd3b9b"

mkdir -p "$(dirname "$MONITOR_LOG")"
printf '[monitor-start] %s waiting_for=exp2_face_pretrained_head32_regression_20frame\n' \
    "$(date -Is)" >> "$MONITOR_LOG"
while pgrep -f '[s]tudy.exp2_face_pretrained_head32_regression.run_all' >/dev/null; do
    sleep 30
done

actual_lab_sha256="$(sha256sum "$ROOT/merged_lab_tests.csv" | awk '{print $1}')"
if [[ "$actual_lab_sha256" != "$EXPECTED_LAB_SHA256" ]]; then
    printf '[monitor-abort] %s merged lab hash changed: %s\n' \
        "$(date -Is)" "$actual_lab_sha256" >> "$MONITOR_LOG"
    exit 1
fi

if ! python - "$UPSTREAM" "$REFERENCE" >> "$MONITOR_LOG" 2>&1 <<'PY'
import hashlib
from pathlib import Path
import sys

import pandas as pd

upstream, reference = map(Path, sys.argv[1:])
run_index = pd.read_csv(upstream / "run_index.csv")
failures = pd.read_csv(upstream / "failures.csv")
expected = {
    (architecture, target)
    for architecture in ("mobilenet_v3_small", "efficientnet_b0")
    for target in ("hemoglobin_low", "po2_low", "oxyhemoglobin_fraction")
}
observed = set(zip(run_index["architecture"], run_index["target"]))
if observed != expected or not run_index["status"].eq("ok").all() or len(failures):
    raise SystemExit(
        f"upstream incomplete: observed={sorted(observed)} failures={len(failures)}"
    )

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for target in ("hemoglobin_low", "po2_low", "oxyhemoglobin_fraction"):
    current = upstream / "task_records" / f"{target}.csv"
    expected_path = reference / "task_records" / f"{target}.csv"
    if digest(current) != digest(expected_path):
        raise SystemExit(f"task record mismatch: {target}")
if digest(upstream / "target_scalers.json") != digest(reference / "target_scalers.json"):
    raise SystemExit("target scaler mismatch")
print("[monitor-validate] upstream complete; records and scalers exactly aligned")
PY
then
    printf '[monitor-abort] %s upstream validation failed\n' "$(date -Is)" >> "$MONITOR_LOG"
    exit 1
fi

printf '[monitor-launch] %s starting=exp2_history_only_head32_regression\n' \
    "$(date -Is)" >> "$MONITOR_LOG"
cd "$ROOT"
bash study/exp2_history_only_head32_regression/launch_screen.sh --overwrite \
    >> "$MONITOR_LOG" 2>&1
printf '[monitor-complete] %s launch submitted\n' "$(date -Is)" >> "$MONITOR_LOG"
