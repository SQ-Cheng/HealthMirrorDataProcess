#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="exp2_sota_ablation"
LOGS="$ROOT/study/exp2_sota_ablation/logs"
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -euo pipefail
  cd "$1"
  exec > >(tee -a "$2/run.log") 2>&1
  printf "[suite-start] %s seven ordered ablations; no old model retraining\n" "$(date -Is)"
  bash study/exp2_sota_ablation/run_all.sh --devices 0,1,2,3,4,5,6,7
  printf "[suite-complete] %s\n" "$(date -Is)"
' _ "$ROOT" "$LOGS"
echo "Detached screen: $SESSION"
echo "Log: $LOGS/run.log"
