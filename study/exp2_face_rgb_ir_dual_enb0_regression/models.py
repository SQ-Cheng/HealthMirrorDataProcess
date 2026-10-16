"""Independent ImageNet-initialized trunks; concat2560 -> head32 -> scalar."""

import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model, SingleTaskHead


class DualEfficientNet(nn.Module):
    def __init__(self, rgb, ir):
        super().__init__()
        self.rgb_encoder = nn.Sequential(rgb.features, rgb.avgpool, nn.Flatten(1))
        self.ir_encoder = nn.Sequential(ir.features, ir.avgpool, nn.Flatten(1))
        self.classifier = SingleTaskHead(2560, hidden_features=32)

    def forward(self, images):
        rgb, ir = images.split(3, dim=1)
        return self.classifier(torch.cat((self.rgb_encoder(rgb), self.ir_encoder(ir)), dim=1))


def build_model(architecture, weights_dir):
    if architecture != "efficientnet_b0":
        raise ValueError("Only independent dual EfficientNet-B0 encoders are supported")
    rgb, _, path = build_pretrained_model(architecture, weights_dir)
    ir, _, ir_path = build_pretrained_model(architecture, weights_dir)
    if path != ir_path:
        raise ValueError("RGB and IR must use the same ImageNet initialization")
    model = DualEfficientNet(rgb, ir)
    return model, model.classifier, path
