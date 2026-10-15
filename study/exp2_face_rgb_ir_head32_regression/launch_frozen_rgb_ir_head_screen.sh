#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SESSION_NAME="${SESSION_NAME:-exp2_rgb_ir_frozen_rgb_ir_head}"
export LOG_FILE="$HERE/logs/rgb_main_frozen_ir_head/run.log"
bash "$HERE/launch_screen.sh" --initialization rgb_main --training-protocol joint_ir_head --no-compile "$@"
