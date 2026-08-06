"""ImageNet-pretrained EfficientNet-B0 recovery regressor."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from .config import HEAD_HIDDEN_FEATURES, PRETRAINED_WEIGHT_FILE, WEIGHTS_DIR


class RecoveryHead(nn.Sequential):
    def __init__(self, in_features):
        super().__init__(
            nn.Linear(in_features, HEAD_HIDDEN_FEATURES),
            nn.LayerNorm(HEAD_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(HEAD_HIDDEN_FEATURES, 1),
            nn.Sigmoid(),
        )


def build_model():
    weight_path = Path(WEIGHTS_DIR) / PRETRAINED_WEIGHT_FILE
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing pretrained weights: {weight_path}")
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    in_features = model.classifier[1].in_features
    model.classifier = RecoveryHead(in_features)
    return model, model.classifier, weight_path


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_stage(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.features[-2], model.features[-1], model.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
