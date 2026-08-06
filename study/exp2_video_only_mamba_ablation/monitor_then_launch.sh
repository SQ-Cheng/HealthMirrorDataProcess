#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/HealthMirrorDataProcess"
PARENT_SESSION="exp2_video_ecg_mamba"
ABLATION_SESSION="exp2_video_only_mamba_ablation"
PARENT_OUTPUT="${ROOT_DIR}/study/exp2_video_ecg_mamba/outputs"
ABLATION_OUTPUT="${ROOT_DIR}/study/exp2_video_only_mamba_ablation/outputs"
LAUNCH_SCRIPT="${ROOT_DIR}/study/exp2_video_only_mamba_ablation/launch_screen.sh"
POLL_SECONDS="${POLL_SECONDS:-60}"

timestamp() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

echo "[$(timestamp)] monitoring parent screen: ${PARENT_SESSION}"
while screen -ls 2>/dev/null | grep -q "[.]${PARENT_SESSION}[[:space:]]"; do
    sleep "${POLL_SECONDS}"
done
echo "[$(timestamp)] parent screen ended; validating completion"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate healthmirrorenv
python - "${PARENT_OUTPUT}" <<'PY'
import os
import sys
import pandas as pd

output = sys.argv[1]
path = os.path.join(output, "run_index.csv")
if not os.path.isfile(path):
    raise SystemExit("Parent run_index.csv is missing; ablation will not launch")
frame = pd.read_csv(path)
expected = {
    "hemoglobin_low",
    "pco2_low",
    "po2_low",
    "high_blood_pressure",
    "lactate_high",
}
successful = set(frame.loc[frame["status"].eq("ok"), "target"])
non_ok = frame.loc[frame["status"].ne("ok"), ["target", "status", "reason"]]
missing = sorted(expected - successful)
duplicates = frame.loc[frame["target"].duplicated(keep=False), "target"].tolist()
if missing or not non_ok.empty or duplicates:
    raise SystemExit(
        f"Parent did not complete successfully: missing={missing}, "
        f"non_ok={non_ok.to_dict('records')}, duplicates={duplicates}"
    )
print("Parent completion validated:", ",".join(sorted(successful)))
PY

if screen -ls 2>/dev/null | grep -q "[.]${ABLATION_SESSION}[[:space:]]"; then
    echo "[$(timestamp)] ablation screen already exists; refusing duplicate launch"
    exit 1
fi
if [[ -e "${ABLATION_OUTPUT}/run_index.csv" ]]; then
    echo "[$(timestamp)] ablation output already contains a run; refusing overwrite"
    exit 1
fi

echo "[$(timestamp)] launching controlled video-only ablation"
bash "${LAUNCH_SCRIPT}"
echo "[$(timestamp)] launch command completed"
