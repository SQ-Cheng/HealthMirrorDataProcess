#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION_NAME:-face224_ir_full_c075}"
LOG_DIR="$ROOT/preprocess/face_crop_comparison/logs/full_ir"
mkdir -p "$LOG_DIR"
for argument in "$@"; do
    case "$argument" in
        --raw-root*|--output-dir*|--video-id*|--modality*)
            echo "This full-data chain uses the fixed original root and IR modality; run individual modules for custom subsets." >&2
            exit 2
            ;;
    esac
done
if screen -list 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    printf 'Screen already exists: %s\n' "$SESSION" >&2
    exit 1
fi
screen -dmS "$SESSION" bash -c '
    ROOT=$1
    LOG_DIR=$2
    shift 2
    cd "$ROOT" || exit 1
    set -o pipefail
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
    PYTHON=/root/miniconda3/envs/healthmirrorenv/bin/python
    printf "[ir-chain-start] %s\n" "$(date -Is)" | tee -a "$LOG_DIR/run.log"
    "$PYTHON" -u -m preprocess.face_crop_comparison.count_retained_rgb_ir_pairs --preflight 2>&1 | tee -a "$LOG_DIR/run.log" || exit 1
    "$PYTHON" -u -m preprocess.face_crop_comparison.process_full_dataset --modality ir --workers 8 "$@" 2>&1 | tee -a "$LOG_DIR/run.log"
    processing_status=${PIPESTATUS[0]}
    printf "[ir-processing-exit] status=%s\n" "$processing_status" | tee -a "$LOG_DIR/run.log"
    # Even terminal source failures must be accounted for by the paired audit.
    "$PYTHON" -u -m preprocess.face_crop_comparison.count_retained_rgb_ir_pairs --workers 4 2>&1 | tee -a "$LOG_DIR/run.log"
    statistics_status=${PIPESTATUS[0]}
    printf "[ir-statistics-exit] status=%s\n" "$statistics_status" | tee -a "$LOG_DIR/run.log"
    if [[ "$processing_status" -ne 0 ]]; then exit "$processing_status"; fi
    exit "$statistics_status"
' _ "$ROOT" "$LOG_DIR" "$@"
printf 'Started: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$LOG_DIR" "$SESSION"
