"""Load the pinned official architecture and refuse unverified formal weights."""

import os
import subprocess
import sys

import requests
import torch
from torch import nn

from . import config


def ensure_source():
    if not config.REPO.exists():
        config.REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", config.REPO_URL, str(config.REPO)], check=True)
        subprocess.run(["git", "-C", str(config.REPO), "checkout", "--detach", config.REPO_REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(config.REPO), "rev-parse", "HEAD"], text=True).strip()
    if actual != config.REPO_REVISION:
        raise RuntimeError("Local official DINOv3 source revision changed")


def weight_sha256():
    if not config.WEIGHTS.is_file():
        raise FileNotFoundError(f"Authorized official DINOv3-S checkpoint is missing: {config.WEIGHTS}")
    digest = config.reference.sha256(config.WEIGHTS)
    if not digest.startswith(config.WEIGHT_HASH_PREFIX):
        raise RuntimeError("DINOv3 weights do not match the official ViT-S/16 LVD1689M hash prefix")
    return digest


def load_encoder(*, pretrained=True):
    ensure_source()
    if pretrained:
        weight_sha256()
    sys.path.insert(0, str(config.REPO))
    from dinov3.hub.backbones import dinov3_vits16
    model = dinov3_vits16(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(config.WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


class Head(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(384, 32), nn.LayerNorm(32), nn.SiLU(),
                         nn.Dropout(config.DROPOUT), nn.Linear(32, 1))


class FrozenPredictor(nn.Module):
    """Online equivalent used to verify that only the head receives gradients."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.head = Head()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)
        return self.head(features)


def main():
    ensure_source()
    url = os.environ.get("DINOV3_WEIGHTS_URL")
    if not config.WEIGHTS.exists() and url:
        config.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "dl.fbaipublicfiles.com":
            raise ValueError("Use an authorized official Meta HTTPS checkpoint URL")
        temporary = config.WEIGHTS.with_suffix(".download")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
        if not config.reference.sha256(temporary).startswith(config.WEIGHT_HASH_PREFIX):
            temporary.unlink()
            raise RuntimeError("Downloaded checkpoint failed the official hash check")
        temporary.replace(config.WEIGHTS)
    print(f"[weights-verified] sha256={weight_sha256()}")


if __name__ == "__main__":
    main()
