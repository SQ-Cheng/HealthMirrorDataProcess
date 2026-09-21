"""Independent pre/post EfficientNet-B0 encoders with the Exp5 fusion head."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from .config import ENCODER_FEATURES, HEAD_HIDDEN_FEATURES, PRETRAINED_WEIGHT_FILE, WEIGHTS_DIR


def _projector(backbone_features):
    return nn.Sequential(
        nn.Linear(backbone_features, ENCODER_FEATURES),
        nn.LayerNorm(ENCODER_FEATURES), nn.SiLU(inplace=True), nn.Dropout(0.10),
    )


def _fusion_head():
    return nn.Sequential(
        nn.Linear(ENCODER_FEATURES * 4, HEAD_HIDDEN_FEATURES),
        nn.LayerNorm(HEAD_HIDDEN_FEATURES), nn.SiLU(inplace=True), nn.Dropout(0.25),
        nn.Linear(HEAD_HIDDEN_FEATURES, 1), nn.Sigmoid(),
    )


class PairedCasusModel(nn.Module):
    def __init__(self, pre_backbone, post_backbone, backbone_features):
        super().__init__()
        self.pre_backbone = pre_backbone
        self.post_backbone = post_backbone
        self.pre_projector = _projector(backbone_features)
        self.post_projector = _projector(backbone_features)
        self.head = _fusion_head()

    @staticmethod
    def _encode(images, backbone, projector):
        features = backbone.features(images)
        return projector(backbone.avgpool(features).flatten(1))

    def forward(self, pre_images, post_images):
        pre = self._encode(pre_images, self.pre_backbone, self.pre_projector)
        post = self._encode(post_images, self.post_backbone, self.post_projector)
        return self.head(torch.cat((pre, post, post - pre, torch.abs(post - pre)), dim=1))


class PostOnlyCasusModel(nn.Module):
    """Capacity-controlled postoperative-only protocol."""

    def __init__(self, backbone, backbone_features):
        super().__init__()
        self.backbone = backbone
        self.projector = _projector(backbone_features)
        self.head = _fusion_head()

    def forward(self, images):
        features = self.backbone.features(images)
        post = self.projector(self.backbone.avgpool(features).flatten(1))
        pre = torch.zeros_like(post)
        return self.head(torch.cat((pre, post, post - pre, torch.abs(post - pre)), dim=1))


def _pretrained_backbone():
    weight_path = Path(WEIGHTS_DIR) / PRETRAINED_WEIGHT_FILE
    if not weight_path.is_file():
        raise FileNotFoundError(f"Missing pretrained weights: {weight_path}")
    backbone = efficientnet_b0(weights=None)
    backbone.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return backbone, features, weight_path


def build_model():
    pre, features, path = _pretrained_backbone()
    post, post_features, _ = _pretrained_backbone()
    if features != post_features:
        raise AssertionError("Pre/post backbone feature dimensions differ")
    return PairedCasusModel(pre, post, features), path


def build_post_only_model():
    backbone, features, path = _pretrained_backbone()
    return PostOnlyCasusModel(backbone, features), path


def backbone_modules(model):
    if isinstance(model, PairedCasusModel):
        return model.pre_backbone, model.post_backbone
    return (model.backbone,)


def projector_modules(model):
    if isinstance(model, PairedCasusModel):
        return model.pre_projector, model.post_projector
    return (model.projector,)


def train_head_modules(model):
    for module in (*projector_modules(model), model.head):
        module.train()


def freeze_backbone(model):
    for backbone in backbone_modules(model):
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    for module in (*projector_modules(model), model.head):
        for parameter in module.parameters():
            parameter.requires_grad = True


def unfreeze_last_stage(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for backbone in backbone_modules(model):
        for module in (backbone.features[-2], backbone.features[-1]):
            for parameter in module.parameters():
                parameter.requires_grad = True
    for module in (*projector_modules(model), model.head):
        for parameter in module.parameters():
            parameter.requires_grad = True


def last_stage_parameters(model):
    return [
        parameter
        for backbone in backbone_modules(model)
        for module in (backbone.features[-2], backbone.features[-1])
        for parameter in module.parameters()
    ]


def head_parameters(model):
    return [
        parameter
        for module in (*projector_modules(model), model.head)
        for parameter in module.parameters()
    ]


def parameter_counts(model):
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )
