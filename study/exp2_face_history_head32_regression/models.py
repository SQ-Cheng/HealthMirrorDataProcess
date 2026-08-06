"""ImageNet backbones fused with a lightweight prior-lab encoder."""

import os

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, mobilenet_v3_small

from .config import (
    HEAD_HIDDEN_FEATURES,
    HISTORY_HIDDEN_FEATURES,
    HISTORY_INPUT_FEATURES,
    HISTORY_OUTPUT_FEATURES,
)


WEIGHT_FILES = {
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
}


class HistoryEncoder(nn.Module):
    """DeepSets-style encoder over all prior measurements."""

    def __init__(self):
        super().__init__()
        self.measurement_mlp = nn.Sequential(
            nn.Linear(HISTORY_INPUT_FEATURES, HISTORY_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Linear(HISTORY_HIDDEN_FEATURES, HISTORY_OUTPUT_FEATURES),
            nn.LayerNorm(HISTORY_OUTPUT_FEATURES),
            nn.SiLU(inplace=True),
        )

    def forward(self, history, history_mask):
        encoded = self.measurement_mlp(history)
        mask = history_mask.unsqueeze(-1).to(encoded.dtype)
        count = mask.sum(dim=1).clamp_min_(1.0)
        return (encoded * mask).sum(dim=1) / count


class FusionHead(nn.Module):
    def __init__(self, image_features, dropout=0.25):
        super().__init__()
        self.history_encoder = HistoryEncoder()
        self.regressor = nn.Sequential(
            nn.Linear(
                image_features + HISTORY_OUTPUT_FEATURES,
                HEAD_HIDDEN_FEATURES,
            ),
            nn.LayerNorm(HEAD_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN_FEATURES, 1),
        )

    def forward(self, image_features, history, history_mask):
        history_features = self.history_encoder(history, history_mask)
        return self.regressor(torch.cat((image_features, history_features), dim=1))


class ImageHistoryRegressor(nn.Module):
    def __init__(self, backbone, image_features):
        super().__init__()
        self.backbone = backbone
        self.fusion = FusionHead(image_features)

    def forward(self, images, history, history_mask):
        image_features = self.backbone.features(images)
        image_features = self.backbone.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        return self.fusion(image_features, history, history_mask)


def build_pretrained_model(architecture, weights_dir):
    """Load the local ImageNet backbone and attach history-aware regression."""
    if architecture not in WEIGHT_FILES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    weight_path = os.path.join(weights_dir, WEIGHT_FILES[architecture])
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Missing pretrained weights: {weight_path}. Run download_weights.py first."
        )

    if architecture == "mobilenet_v3_small":
        backbone = mobilenet_v3_small(weights=None)
        image_features = backbone.classifier[0].in_features
    else:
        backbone = efficientnet_b0(weights=None)
        image_features = backbone.classifier[1].in_features
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    backbone.classifier = nn.Identity()
    model = ImageHistoryRegressor(backbone, image_features)
    return model, model.fusion, weight_path


def freeze_encoder(model, fusion):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in fusion.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def history_encoder_parameter_count(model):
    return sum(parameter.numel() for parameter in model.fusion.history_encoder.parameters())
