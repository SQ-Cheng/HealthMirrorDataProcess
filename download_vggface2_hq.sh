#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$(realpath "$0")"
ROOT="$(dirname "$SCRIPT")"
DEST="$HOME/shared/VGGFace2-HQ"
ARCHIVES="$DEST/archives"
IMAGES="$DEST/images"
LOGS="$DEST/logs"
PYTHON="/root/miniconda3/envs/healthmirrorenv/bin/python"
HF="/root/miniconda3/envs/healthmirrorenv/bin/hf"
REVISION="6771c0319f29a7b77e4f71ebfb09f7b40066e2cc"
SESSION="download_vggface2_hq"
mkdir -p "$LOGS"

if [[ "${1:-}" != "--worker" ]]; then
  if screen -list | grep -q "[.]$SESSION[[:space:]]"; then
    echo "Screen session already exists: $SESSION" >&2
    exit 1
  fi
  screen -dmS "$SESSION" bash -c '
    set -o pipefail
    bash "$1" --worker 2>&1 | tee -a "$2/run.log"
    status=$?
    printf "[screen-exit] status=%s\n" "$status" | tee -a "$2/run.log"
    exit "$status"
  ' _ "$SCRIPT" "$LOGS"
  printf 'Detached screen: %s\nDestination: %s\nLog: %s/run.log\nAttach: screen -r %s\n' "$SESSION" "$DEST" "$LOGS" "$SESSION"
  exit 0
fi

export TZ=Asia/Shanghai
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 TQDM_MININTERVAL=15
exec 9>"$DEST/.download.lock"
flock -n 9
trap 'status=$?; if [[ $status -ne 0 ]]; then printf "[failed] %s status=%s; rerun launcher to resume\n" "$(date -Is)" "$status"; fi' EXIT

if [[ -f "$DEST/COMPLETE" ]]; then
  echo "[complete] Download and extraction already finished: $DEST"
  exit 0
fi
mkdir -p "$ARCHIVES" "$IMAGES"
cp "$ROOT/download_vggface2_hq.sha256" "$DEST/SHA256SUMS"
printf '[download-start] %s repo=RichardErkhov/VGGFace2-HQ revision=%s total_bytes=95708719289\n' "$(date -Is)" "$REVISION"
downloaded=0
for attempt in {1..8}; do
  if "$HF" download RichardErkhov/VGGFace2-HQ \
    original/VGGface2_HQ.z01 original/VGGface2_HQ.z02 \
    original/VGGface2_HQ.z03 original/VGGface2_HQ.z04 original/VGGface2_HQ.zip \
    --repo-type dataset --revision "$REVISION" --local-dir "$ARCHIVES" --max-workers 2; then
    downloaded=1
    break
  fi
  printf '[retry] %s attempt=%s/8; partial downloads retained\n' "$(date -Is)" "$attempt"
  if [[ "$attempt" != 8 ]]; then sleep 60; fi
done
[[ "$downloaded" == 1 ]] || exit 1

cd "$ARCHIVES/original"
printf '[checksum-start] %s\n' "$(date -Is)"
sha256sum -c "$DEST/SHA256SUMS"
printf '[archive-check] %s inspecting paths and available extraction space\n' "$(date -Is)"
7z l -slt -ba VGGface2_HQ.zip | "$PYTHON" -c '
import shutil, sys
from pathlib import PurePosixPath
total = count = 0
for line in sys.stdin:
    if line.startswith("Path = "):
        name = line.split(" = ", 1)[1].rstrip("\r\n").replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or (len(name) > 1 and name[1] == ":"):
            raise RuntimeError(f"Unsafe archive path: {name!r}")
        count += 1
    elif line.startswith(("Symbolic Link = ", "Hard Link = ")):
        raise RuntimeError("Archive links are not expected in an image dataset")
    elif line.startswith("Size = "):
        total += int(line.split(" = ", 1)[1])
free = shutil.disk_usage(sys.argv[1]).free
print(f"[archive-index] entries={count} uncompressed_bytes={total} available_bytes={free}", flush=True)
if not count or free < total + 1024**3:
    raise RuntimeError("Empty archive or insufficient extraction space")
' "$IMAGES"
printf '[extract-start] %s destination=%s\n' "$(date -Is)" "$IMAGES"
7z x -y -mmt=4 -bso0 -bse1 -bsp1 "-o$IMAGES" VGGface2_HQ.zip
touch "$DEST/COMPLETE"
printf '\n[complete] %s images=%s archives=%s\n' "$(date -Is)" "$IMAGES" "$ARCHIVES/original"
