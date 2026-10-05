"""Download the official MediaPipe detector for the active comparison."""

import hashlib
import json
from pathlib import Path
import urllib.request


MODEL_DIR = Path(__file__).resolve().parent / "models"
MP_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def download(url, destination):
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"[download] {url}", flush=True)
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mp_path = MODEL_DIR / "blaze_face_short_range.tflite"
    if not mp_path.exists():
        download(MP_URL, mp_path)
    manifest = {
        "mediapipe": {"path": mp_path.name, "source": MP_URL},
    }
    for model in manifest.values():
        path = MODEL_DIR / model["path"]
        model["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        model["size_bytes"] = path.stat().st_size
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[models-ready]", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
