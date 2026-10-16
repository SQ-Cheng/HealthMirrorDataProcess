#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION=exp2_sota_5fold_ablation
LOGS="$ROOT/study/$SESSION/logs"
DEVICES="${1:-0,1,2,3,4,5,6,7}"
if (($#)); then
    shift
fi
EXTRA=("$@")
PAUSED="$ROOT/study/$SESSION/outputs/paused_variants.json"
if ((${#EXTRA[@]} == 0)) && [[ -f "$PAUSED" ]]; then
    cd "$ROOT"
    mapfile -t SELECTED < <(/root/miniconda3/envs/healthmirrorenv/bin/python -c '
import json, sys
from study.exp2_sota_5fold_ablation.config import VARIANTS
paused = set(json.load(open(sys.argv[1]))["paused_variants"])
unknown = paused - set(VARIANTS)
if unknown:
    raise ValueError(f"Unknown paused variants: {unknown}")
print("\n".join(name for name in VARIANTS if name not in paused))
' "$PAUSED")
    if ((${#SELECTED[@]} == 0)); then
        echo "No runnable variants; inspect $PAUSED" >&2
        exit 1
    fi
    EXTRA=(--variants "${SELECTED[@]}")
fi
if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
    echo "Screen already exists: $SESSION" >&2
    exit 1
fi
mkdir -p "$LOGS"
screen -dmS "$SESSION" bash -c '
  set -euo pipefail
  cd "$1"
  exec > >(tee -a "$2/run.log") 2>&1
  printf "[suite-start] %s continuous five-fold queue; selected arguments:" "$(date -Is)"
  printf " %q" "${@:4}"
  printf "\n"
  bash study/exp2_sota_5fold_ablation/run_all.sh --devices "$3" "${@:4}"
' _ "$ROOT" "$LOGS" "$DEVICES" "${EXTRA[@]}"
echo "Detached screen: $SESSION"
echo "Log: $LOGS/run.log"
