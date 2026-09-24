"""Shared or independent EfficientNet encoders for face-pair regression."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from .config import HEAD_HIDDEN_FEATURES, PRETRAINED_WEIGHT_FILE, WEIGHTS_DIR


class PairedDeltaModel(nn.Module):
    def __init__(self, backbone, feature_count):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feature_count, HEAD_HIDDEN_FEATURES),
            nn.LayerNorm(HEAD_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(HEAD_HIDDEN_FEATURES, 1),
        )

    def encode(self, images):
        features = self.backbone.features(images)
        return self.backbone.avgpool(features).flatten(1)

    def forward(self, first, second):
        count = len(first)
        features = self.encode(torch.cat((first, second), dim=0))
        first_features, second_features = features[:count], features[count:]
        return self.head(second_features - first_features)


class IndependentPairedDeltaModel(nn.Module):
    """Separate pretrained parameters for early and late face timepoints."""

    def __init__(self, first_backbone, second_backbone, feature_count):
        super().__init__()
        self.first_backbone = first_backbone
        self.second_backbone = second_backbone
        self.head = nn.Sequential(
            nn.Linear(feature_count, HEAD_HIDDEN_FEATURES),
            nn.LayerNorm(HEAD_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(HEAD_HIDDEN_FEATURES, 1),
        )

    @staticmethod
    def encode(images, backbone):
        features = backbone.features(images)
        return backbone.avgpool(features).flatten(1)

    def forward(self, first, second):
        first_features = self.encode(first, self.first_backbone)
        second_features = self.encode(second, self.second_backbone)
        return self.head(second_features - first_features)


def _pretrained_backbone(weight_path):
    # Loading a fixed pretrained encoder should not perturb head/dropout RNG.
    with torch.random.fork_rng(devices=[]):
        backbone = efficientnet_b0(weights=None)
    backbone.load_state_dict(
        torch.load(weight_path, map_location="cpu", weights_only=True), strict=True
    )
    feature_count = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return backbone, feature_count


def build_model(variant="shared"):
    weight_path = Path(WEIGHTS_DIR) / PRETRAINED_WEIGHT_FILE
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing pretrained weights: {weight_path}")
    first_backbone, feature_count = _pretrained_backbone(weight_path)
    if variant == "shared":
        model = PairedDeltaModel(first_backbone, feature_count)
    elif variant == "independent_backbones":
        second_backbone, second_feature_count = _pretrained_backbone(weight_path)
        if second_feature_count != feature_count:
            raise AssertionError("Early/late backbone feature dimensions differ")
        model = IndependentPairedDeltaModel(
            first_backbone, second_backbone, feature_count
        )
    else:
        raise ValueError(f"Unsupported model variant: {variant}")
    return model, weight_path


def backbone_modules(model):
    if isinstance(model, IndependentPairedDeltaModel):
        return (model.first_backbone, model.second_backbone)
    return (model.backbone,)


def freeze_backbone(model):
    for backbone in backbone_modules(model):
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def parameter_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
