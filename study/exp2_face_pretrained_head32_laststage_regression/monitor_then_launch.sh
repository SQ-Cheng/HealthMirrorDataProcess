#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
PREVIOUS_SESSION="${PREVIOUS_SESSION:-exp2_face_pretrained_head64_regression_20frame}"
NEXT_SESSION="${NEXT_SESSION:-exp2_face_pretrained_head32_laststage_regression_20frame}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PREVIOUS_OUTPUT="${ROOT_DIR}/study/exp2_face_pretrained_head64_regression/outputs/20frame"
NEXT_LAUNCHER="${ROOT_DIR}/study/exp2_face_pretrained_head32_laststage_regression/launch_screen.sh"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] monitoring ${PREVIOUS_SESSION}"
poll_count=0
while screen -ls 2>/dev/null | grep -q "[.]${PREVIOUS_SESSION}[[:space:]]"; do
    sleep "${POLL_SECONDS}"
    poll_count=$((poll_count + 1))
    if (( poll_count % 10 == 0 )); then
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ${PREVIOUS_SESSION} still running"
    fi
done

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] validating Head64 regression"
/root/miniconda3/envs/healthmirrorenv/bin/python - "${PREVIOUS_OUTPUT}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

output = Path(sys.argv[1])
run_index_path = output / "run_index.csv"
if not run_index_path.is_file():
    raise SystemExit(f"Head64 regression has no run index: {run_index_path}")
run_index = pd.read_csv(run_index_path)
expected = {
    (architecture, target)
    for architecture in ("mobilenet_v3_small", "efficientnet_b0")
    for target in ("hemoglobin_low", "po2_low", "lactate_high")
}
successful = set(
    run_index.loc[run_index["status"].eq("ok"), ["architecture", "target"]]
    .itertuples(index=False, name=None)
)
if successful != expected:
    raise SystemExit(f"Head64 regression incomplete: missing={expected-successful}")
failures = output / "failures.csv"
if not failures.is_file() or not pd.read_csv(failures).empty:
    raise SystemExit("Head64 regression contains failed jobs")
required = (
    "training_curves.png",
    "test_metrics.png",
    "split_generalization.png",
    "test_predicted_vs_true.png",
)
missing = [name for name in required if not (output / "figures" / name).is_file()]
if missing:
    raise SystemExit(f"Head64 regression plotting incomplete: {missing}")
print("Head64 regression completed all six jobs and result plotting")
PY

if screen -ls 2>/dev/null | grep -q "[.]${NEXT_SESSION}[[:space:]]"; then
    echo "Next experiment screen already exists: ${NEXT_SESSION}" >&2
    exit 1
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] launching ${NEXT_SESSION}"
SESSION_NAME="${NEXT_SESSION}" bash "${NEXT_LAUNCHER}" 20frame --overwrite
