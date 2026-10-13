#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION_NAME:-exp2_rgb_ir_head32_regression}"
LOG="$ROOT/study/exp2_face_rgb_ir_head32_regression/logs/run.log"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$(dirname "$LOG")"
printf -v OPTIONS ' %q' "$@"
screen -dmS "$SESSION" bash -lc "set -o pipefail; cd '$ROOT' && source /root/miniconda3/etc/profile.d/conda.sh && conda activate healthmirrorenv && export HEALTHMIRROR_FACE_SOURCE=face224 PYTHONUNBUFFERED=1 MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2 && python -u -m study.exp2_face_rgb_ir_head32_regression.run$OPTIONS 2>&1 | tee '$LOG'"
echo "Screen: $SESSION"
echo "Attach: screen -r $SESSION"
echo "Log: $LOG"
