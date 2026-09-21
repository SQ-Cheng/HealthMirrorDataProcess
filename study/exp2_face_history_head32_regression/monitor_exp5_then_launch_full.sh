#!/usr/bin/env bash
set -euo pipefail

root="/root/autodl-tmp/HealthMirrorDataProcess"
current_log="$root/study/exp5_face_pair_casus/logs/post_face_only_run.log"
monitor_log="$root/study/exp2_face_history_head32_regression/logs/monitor_exp5_then_full.log"
expected_lab_sha256="5e41a342586f85fe354ac668f23b1c3072790f560e2b6a2267bf71377dcd3b9b"

mkdir -p "$(dirname "$monitor_log")"
printf '[monitor-start] %s waiting_for=exp5_casus_post_face_only\n' "$(date -Is)" >> "$monitor_log"
while pgrep -f "[s]tudy.exp5_face_pair_casus.run_post_face_only" >/dev/null; do
  sleep 30
done

if ! grep -q '^\[experiment-complete\]' "$current_log"; then
  printf '[monitor-abort] %s upstream did not complete successfully\n' "$(date -Is)" >> "$monitor_log"
  exit 1
fi

actual_lab_sha256="$(sha256sum "$root/merged_lab_tests.csv" | awk '{print $1}')"
if [[ "$actual_lab_sha256" != "$expected_lab_sha256" ]]; then
  printf '[monitor-abort] %s merged lab hash changed: %s\n' "$(date -Is)" "$actual_lab_sha256" >> "$monitor_log"
  exit 1
fi

printf '[monitor-launch] %s starting union-data Exp2 face-history regression\n' "$(date -Is)" >> "$monitor_log"
cd "$root"
bash study/exp2_face_history_head32_regression/launch_screen.sh \
  --overwrite --rebuild-split
printf '[monitor-complete] %s launch command returned\n' "$(date -Is)" >> "$monitor_log"
