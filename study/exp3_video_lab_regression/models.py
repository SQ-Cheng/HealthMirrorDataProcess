"""Kinetics-pretrained R3D-18 encoder and a 32-unit regression head."""

import hashlib

import torch
from torch import nn
from torchvision.models.video import r3d_18

from .config import HEAD_HIDDEN, WEIGHT_PATH


class VideoRegressor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        features = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(features, HEAD_HIDDEN),
            nn.LayerNorm(HEAD_HIDDEN),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, clips):
        return self.head(self.encoder(clips)).squeeze(1)


def build_model():
    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(WEIGHT_PATH)
    with open(WEIGHT_PATH, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if not digest.startswith("b3b3357e"):
        raise ValueError(f"Unexpected R3D-18 weight checksum: {digest}")
    encoder = r3d_18(weights=None)
    encoder.load_state_dict(
        torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True), strict=True
    )
    return VideoRegressor(encoder), digest


def freeze_encoder(model):
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True
