"""ImageNet-pretrained backbones with independent binary classification heads."""

import os

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet18

from .config import HEAD_HIDDEN_FEATURES


WEIGHT_FILES = {
    "resnet18": "resnet18-f37072fd.pth",
    "mobilenet_v3_small": "mobilenet_v3_small-047dcff4.pth",
    "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
}


class SingleTaskHead(nn.Sequential):
    def __init__(
        self,
        in_features,
        hidden_features=HEAD_HIDDEN_FEATURES,
        dropout=0.25,
    ):
        super().__init__(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )


def build_pretrained_model(architecture, weights_dir):
    """Load a local ImageNet state dict, then replace its classifier."""
    if architecture not in WEIGHT_FILES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    weight_path = os.path.join(weights_dir, WEIGHT_FILES[architecture])
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Missing pretrained weights: {weight_path}. Run download_weights.py first."
        )

    if architecture == "resnet18":
        model = resnet18(weights=None)
    elif architecture == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
    else:
        model = efficientnet_b0(weights=None)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    if architecture == "resnet18":
        in_features = model.fc.in_features
        model.fc = SingleTaskHead(in_features)
        head = model.fc
    elif architecture == "mobilenet_v3_small":
        in_features = model.classifier[0].in_features
        model.classifier = SingleTaskHead(in_features)
        head = model.classifier
    else:
        in_features = model.classifier[1].in_features
        model.classifier = SingleTaskHead(in_features)
        head = model.classifier
    return model, head, weight_path


def freeze_encoder(model, head):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True


def _last_backbone_stage(model, architecture):
    if architecture == "mobilenet_v3_small":
        return model.features[9:]
    if architecture == "efficientnet_b0":
        return model.features[7:]
    if architecture == "resnet18":
        return model.layer4
    raise ValueError(f"Unsupported architecture: {architecture}")


def unfreeze_last_stage(model, head, architecture):
    freeze_encoder(model, head)
    for parameter in _last_backbone_stage(model, architecture).parameters():
        parameter.requires_grad = True


def configure_training_mode(model, head, architecture, scope):
    model.eval()
    if scope == "head":
        head.train()
        return
    if scope == "last_stage":
        _last_backbone_stage(model, architecture).train()
        head.train()
        return
    raise ValueError(f"Unsupported training scope: {scope}")


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
