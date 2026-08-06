#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
SESSION="${SESSION:-exp2_face_pretrained_head32_regression_20frame}"
POLL_SECONDS="${POLL_SECONDS:-60}"
OUTPUT="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/outputs/20frame"
REFERENCE="${ROOT_DIR}/study/exp2_face_history_head32_regression/outputs/20frame"
INDEX="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/cache/20frame_index"
LAUNCHER="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression/launch_screen.sh"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

echo "[$(timestamp)] monitoring ${SESSION}"
while screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; do
    sleep "${POLL_SECONDS}"
done

echo "[$(timestamp)] validating completed Hb/PO2 run"
"${PYTHON}" - "${OUTPUT}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

output = Path(sys.argv[1])
run_index = pd.read_csv(output / "run_index.csv")
expected = {
    (architecture, target)
    for architecture in ("mobilenet_v3_small", "efficientnet_b0")
    for target in ("hemoglobin_low", "po2_low")
}
successful = set(
    run_index.loc[run_index["status"].eq("ok"), ["architecture", "target"]]
    .itertuples(index=False, name=None)
)
if successful != expected or len(run_index) != len(expected):
    raise SystemExit(f"Hb/PO2 run incomplete: {sorted(successful)}")
if not pd.read_csv(output / "failures.csv").empty:
    raise SystemExit("Hb/PO2 run contains failed jobs")
required = {
    "training_curves.png",
    "test_metrics.png",
    "split_generalization.png",
    "test_predicted_vs_true.png",
}
missing = sorted(name for name in required if not (output / "figures" / name).is_file())
if missing:
    raise SystemExit(f"Hb/PO2 figures are incomplete: {missing}")
print("Hb/PO2 completion validated")
PY

echo "[$(timestamp)] adding oxyhemoglobin-fraction regression"
SESSION_NAME="${SESSION}" bash "${LAUNCHER}" 20frame \
    --targets oxyhemoglobin_fraction \
    --add-targets \
    --reference-output-dir "${REFERENCE}" \
    --index-dir "${INDEX}"
