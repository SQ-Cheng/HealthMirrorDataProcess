#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
PREVIOUS_SESSION="${PREVIOUS_SESSION:-exp2_face_history_head32_regression_20frame}"
NEXT_SESSION="${NEXT_SESSION:-exp2_face_pretrained_head32_regression_20frame}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PREVIOUS_OUTPUT="${ROOT_DIR}/study/exp2_face_history_head32_regression/outputs/20frame"
EXPERIMENT_DIR="${ROOT_DIR}/study/exp2_face_pretrained_head32_regression"
CACHE_INDEX="${EXPERIMENT_DIR}/cache/20frame_index"
OLD_INDEX="${EXPERIMENT_DIR}/outputs/20frame/frame_index"
LAUNCHER="${EXPERIMENT_DIR}/launch_screen.sh"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

echo "[$(timestamp)] monitoring ${PREVIOUS_SESSION}"
while screen -ls 2>/dev/null | grep -q "[.]${PREVIOUS_SESSION}[[:space:]]"; do
    sleep "${POLL_SECONDS}"
done

echo "[$(timestamp)] validating completed face-plus-history experiment"
"${PYTHON}" - "${PREVIOUS_OUTPUT}" <<'PY'
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
    raise SystemExit(
        f"History experiment incomplete: expected={sorted(expected)}, "
        f"successful={sorted(successful)}"
    )
failures = pd.read_csv(output / "failures.csv")
if not failures.empty:
    raise SystemExit("History experiment contains failed jobs")
required = {
    "training_curves.png",
    "test_metrics.png",
    "split_generalization.png",
    "test_predicted_vs_true.png",
}
missing = sorted(name for name in required if not (output / "figures" / name).is_file())
if missing:
    raise SystemExit(f"History experiment figures are incomplete: {missing}")
print("History experiment completion validated")
PY

if [[ ! -s "${CACHE_INDEX}/frame_offsets.npz" ]]; then
    echo "[$(timestamp)] preserving compact 20-frame byte-offset index"
    test -s "${OLD_INDEX}/frame_offsets.npz"
    mkdir -p "${CACHE_INDEX}"
    cp -a "${OLD_INDEX}/." "${CACHE_INDEX}/"
fi

SMOKE_ROOT="$(mktemp -d /tmp/exp2_face_pretrained_raw_smoke.XXXXXX)"
cleanup() {
    rm -rf "${SMOKE_ROOT}"
}
trap cleanup EXIT

echo "[$(timestamp)] running two-target MobileNet smoke test"
cd "${ROOT_DIR}"
CUDA_VISIBLE_DEVICES=0,1 MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 \
"${PYTHON}" -u -m study.exp2_face_pretrained_head32_regression.run_all \
    --frame-policy 20frame \
    --output-dir "${SMOKE_ROOT}/outputs" \
    --source-dir "${SMOKE_ROOT}/source_data" \
    --index-dir "${CACHE_INDEX}" \
    --reference-output-dir "${PREVIOUS_OUTPUT}" \
    --architectures mobilenet_v3_small \
    --targets hemoglobin_low,po2_low \
    --smoke-test --overwrite

"${PYTHON}" - "${SMOKE_ROOT}/outputs" "${PREVIOUS_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

output, reference = map(Path, sys.argv[1:])
run_index = pd.read_csv(output / "run_index.csv")
if len(run_index) != 2 or not run_index["status"].eq("ok").all():
    raise SystemExit("Raw-value baseline smoke test failed")
scalers = json.loads((output / "target_scalers.json").read_text())["targets"]
for target in ("hemoglobin_low", "po2_low"):
    current = pd.read_csv(output / "task_records" / f"{target}.csv").sort_values("video_id")
    prior = pd.read_csv(reference / "task_records" / f"{target}.csv").sort_values("video_id")
    for column in ("video_id", "hospital_id", "split", "raw_value"):
        if not np.array_equal(current[column].to_numpy(), prior[column].to_numpy()):
            raise SystemExit(f"Smoke data mismatch for {target}/{column}")
    scaler = scalers[target]
    expected = (current["raw_value"] - scaler["median"]) / scaler["iqr"]
    if not np.allclose(current["robust_scaled_raw_value"], expected, atol=1e-12):
        raise SystemExit(f"Smoke scaler mismatch for {target}")
print("Raw-value baseline smoke test and exact data alignment passed")
PY

if screen -ls 2>/dev/null | grep -q "[.]${NEXT_SESSION}[[:space:]]"; then
    echo "Next experiment screen already exists: ${NEXT_SESSION}" >&2
    exit 1
fi

echo "[$(timestamp)] launching ${NEXT_SESSION}"
SESSION_NAME="${NEXT_SESSION}" bash "${LAUNCHER}" 20frame \
    --targets hemoglobin_low,po2_low \
    --reference-output-dir "${PREVIOUS_OUTPUT}" \
    --index-dir "${CACHE_INDEX}" \
    --overwrite
