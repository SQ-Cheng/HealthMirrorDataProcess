"""Download and validate the three official torchvision ImageNet checkpoints."""

import hashlib
import json
import os
from urllib.parse import urlparse

import torch
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

from .config import WEIGHTS_DIR
from .models import WEIGHT_FILES, build_pretrained_model


WEIGHTS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "mobilenet_v3_small": MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_all(weights_dir=WEIGHTS_DIR):
    os.makedirs(weights_dir, exist_ok=True)
    rows = []
    for architecture, weights in WEIGHTS.items():
        expected_name = os.path.basename(urlparse(weights.url).path)
        if expected_name != WEIGHT_FILES[architecture]:
            raise RuntimeError(
                f"Weight filename mismatch for {architecture}: {expected_name}"
            )
        print(f"Downloading/verifying {architecture}: {weights.url}", flush=True)
        torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=weights_dir,
            progress=True,
            check_hash=True,
            map_location="cpu",
        )
        path = os.path.join(weights_dir, expected_name)
        model, _, _ = build_pretrained_model(architecture, weights_dir)
        rows.append({
            "architecture": architecture,
            "torchvision_weight": str(weights),
            "source_url": weights.url,
            "local_file": expected_name,
            "size_bytes": os.path.getsize(path),
            "sha256": _sha256(path),
            "strict_load_verified": True,
        })
        del model
    manifest_path = os.path.join(weights_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "weights": rows}, handle, indent=2)
    print(f"Weight manifest saved to {manifest_path}", flush=True)
    return rows


if __name__ == "__main__":
    download_all()
