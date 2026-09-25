#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_LOG="${ROOT_DIR}/study/exp6_face_pair_lab_delta/logs/shared_views3.log"
SOURCE_OUTPUT="${ROOT_DIR}/study/exp6_face_pair_lab_delta/outputs/shared_views3"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"

printf '[monitor-start] waiting_for=exp6_shared_views3 time=%s\n' "$(date -Is)"
while screen -ls 2>/dev/null | grep -q '[.]exp6_shared_views3[[:space:]]'; do
    sleep 30
done

if ! grep -q '^\[experiment-complete\] variant=shared_views3 ' "${SOURCE_LOG}"; then
    printf '[monitor-error] Exp6 views3 ended without completion marker\n' >&2
    exit 1
fi
"${PYTHON}" -c '
import pandas as pd
from pathlib import Path
import sys
root = Path(sys.argv[1])
index = pd.read_csv(root / "run_index.csv")
assert len(index) == 9 and index.status.eq("ok").all(), "Exp6 views3 jobs incomplete"
for name in ("test_performance.png", "training_histories.png", "observed_vs_predicted.png"):
    assert (root / "figures" / name).is_file(), f"Missing Exp6 figure: {name}"
' "${SOURCE_OUTPUT}"

printf '[monitor-ready] Exp6 views3 complete; starting trajectory ablations time=%s\n' "$(date -Is)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1
exec "${PYTHON}" -u -m study.exp2_history_only_head32_regression.trajectory_ablations
