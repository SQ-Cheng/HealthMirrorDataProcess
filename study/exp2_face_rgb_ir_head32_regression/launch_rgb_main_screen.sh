#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SESSION_NAME="${SESSION_NAME:-exp2_rgb_ir_rgb_main_init}"
export LOG_FILE="$HERE/logs/rgb_main_init/run.log"
bash "$HERE/launch_screen.sh" --initialization rgb_main "$@"
