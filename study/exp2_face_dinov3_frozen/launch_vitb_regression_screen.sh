#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_dinov3_vitb_head32_regression"
LOGS="$ROOT/study/exp2_face_dinov3_frozen/logs/main24h_regression_vitb16"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -euo pipefail
  cd "$1"
  export HEALTHMIRROR_FACE_SOURCE=face224 CUDA_VISIBLE_DEVICES=0,1,2,3
  unset HEALTHMIRROR_LAB_RUN_TAG HEALTHMIRROR_LAB_OVERWRITE
  export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python=/root/miniconda3/envs/healthmirrorenv/bin/python
  exec > >(tee -a "$2/run.log") 2>&1
  printf "[queue-start] %s ViT-B/16 head32; waiting for four free GPUs\n" "$(date -Is)"
  while pgrep -f "^(python|/root/miniconda3/envs/healthmirrorenv/bin/python) -u -m study[.]exp2_face_rgb_ir_head32_regression[.]run .*--training-protocol joint_ir_head" >/dev/null ||
    ! "$python" -c "import subprocess,sys; rows=subprocess.check_output([\"nvidia-smi\",\"--query-gpu=index,memory.used,utilization.gpu\",\"--format=csv,noheader,nounits\"],text=True).strip().splitlines(); rows=[list(map(int,row.split(\",\"))) for row in rows]; sys.exit(0 if len(rows)>=4 and all(memory<1024 and util<10 for index,memory,util in rows if index<4) else 1)"; do
    printf "[waiting-gpus] %s existing training is untouched\n" "$(date -Is)"
    sleep 60
  done
  printf "[resources-ready] %s starting frozen features and real-feature smoke\n" "$(date -Is)"
  "$python" -u -m study.exp2_face_dinov3_frozen.run_main_regression --backbone dinov3_vitb16 --head-hidden 32 --smoke
  "$python" -u -m study.exp2_face_dinov3_frozen.run_main_regression --backbone dinov3_vitb16 --head-hidden 32
  printf "[queue-complete] %s ViT-B/16 heads and updated comparison figures completed\n" "$(date -Is)"
' _ "$ROOT" "$LOGS"
echo "Detached screen: $SESSION"
echo "Log: $LOGS/run.log"
