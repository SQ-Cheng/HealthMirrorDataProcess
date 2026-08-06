"""ImageNet-pretrained backbones with one shared five-output regression head."""

import os

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, mobilenet_v3_small

from .config import HEAD_HIDDEN_FEATURES, NUM_OUTPUTS


WEIGHT_FILES = {
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
}


class MultiOutputHead(nn.Sequential):
    def __init__(
        self,
        in_features,
        hidden_features=HEAD_HIDDEN_FEATURES,
        outputs=NUM_OUTPUTS,
        dropout=0.25,
    ):
        super().__init__(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, outputs),
        )


def build_pretrained_model(architecture, weights_dir):
    """Load an unchanged local ImageNet backbone and replace only its classifier."""
    if architecture not in WEIGHT_FILES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    weight_path = os.path.join(weights_dir, WEIGHT_FILES[architecture])
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Missing pretrained weights: {weight_path}")

    if architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
    else:
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.classifier = MultiOutputHead(in_features)
    return model, model.classifier, weight_path


def freeze_encoder(model, head):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
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
