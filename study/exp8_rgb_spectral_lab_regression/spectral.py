"""Frozen, official MST++ RGB-to-visible-spectrum estimator."""

import hashlib

import torch
from torch.nn import functional as F

from .config import FEATURE_SHAPE, SOURCE_IMAGE_SIZE, VARIANT, WEIGHT_PATH
from .vendor.mst_plus_plus import MST_Plus_Plus


EXPECTED_SHA256 = "d285430cc688d08434582eee71bae2d82661be7997af1a68d6636ec25f7a3421"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256():
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"Missing MST++ checkpoint: {WEIGHT_PATH}")
    digest = sha256(WEIGHT_PATH)
    if (VARIANT == "ntire2022" and digest != EXPECTED_SHA256) or (
        VARIANT == "hyperskin" and digest == EXPECTED_SHA256
    ):
        raise RuntimeError(f"Wrong checkpoint for Exp8 variant {VARIANT}: {WEIGHT_PATH}")
    return digest


def load_reconstructor(device):
    checkpoint_sha256()
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)
    model = MST_Plus_Plus()
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(
        {key.removeprefix("module."): value
         for key, value in state_dict.items()}, strict=True
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def reconstruct_features(model, images, device):
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (224, 224):
        raise ValueError(f"Expected native RGB 224x224 batch, got {tuple(images.shape)}")
    rgb = images.to(device, non_blocking=True, dtype=torch.float32)
    minimum = rgb.amin(dim=(1, 2, 3), keepdim=True)
    maximum = rgb.amax(dim=(1, 2, 3), keepdim=True)
    if torch.any(maximum <= minimum):
        raise ValueError("Constant RGB frame cannot be normalized for MST++")
    rgb = (rgb - minimum) / (maximum - minimum)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        cube = model(rgb)
    if cube.shape[1] != FEATURE_SHAPE[0] or not torch.isfinite(cube).all():
        raise RuntimeError("MST++ returned invalid spectral estimates")
    # Keep a coarse spatial layout without persisting full spectral cubes.
    height, width = cube.shape[-2:]
    margin_h, margin_w = height // 8, width // 8
    central = cube.float().clamp(0, 1)[:, :, margin_h:height - margin_h, margin_w:width - margin_w]
    return F.adaptive_avg_pool2d(central, FEATURE_SHAPE[1:]).cpu()
