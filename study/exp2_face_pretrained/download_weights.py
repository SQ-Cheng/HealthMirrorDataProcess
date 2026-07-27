"""Download and verify the shared torchvision ImageNet checkpoints."""

import hashlib
import json
import os
from urllib.parse import urlparse

import torch
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_weights")
WEIGHTS = {
    "resnet18": (ResNet18_Weights.IMAGENET1K_V1, resnet18),
    "mobilenet_v3_small": (
        MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        mobilenet_v3_small,
    ),
    "efficientnet_b0": (EfficientNet_B0_Weights.IMAGENET1K_V1, efficientnet_b0),
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
    for architecture, (weights, constructor) in WEIGHTS.items():
        filename = os.path.basename(urlparse(weights.url).path)
        path = os.path.join(weights_dir, filename)
        print(f"Downloading/verifying {architecture}: {weights.url}", flush=True)
        state_dict = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=weights_dir,
            progress=True,
            check_hash=True,
            map_location="cpu",
        )
        constructor(weights=None).load_state_dict(state_dict, strict=True)
        rows.append(
            {
                "architecture": architecture,
                "torchvision_weight": str(weights),
                "source_url": weights.url,
                "local_file": filename,
                "size_bytes": os.path.getsize(path),
                "sha256": _sha256(path),
                "strict_load_verified": True,
            }
        )
    with open(
        os.path.join(weights_dir, "manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"schema_version": 1, "weights": rows}, handle, indent=2)
    return rows


if __name__ == "__main__":
    download_all()
