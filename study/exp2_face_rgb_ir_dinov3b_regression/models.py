"""Late RGB/IR fusion; the task head always has a hidden width of 32."""

import torch
from torch import nn

from study.exp2_face_dinov3_farl_regression.models import regression_head
from . import config


class ModalityFusion(nn.Module):
    def __init__(self, modality_features, method):
        super().__init__()
        self.modality_features, self.method = modality_features, method
        if method == "concat":
            self.head = regression_head(2 * modality_features)
        elif method in ("vector_gate", "scalar_gate"):
            def projection():
                return nn.Sequential(nn.Linear(modality_features, 64), nn.LayerNorm(64), nn.SiLU())
            self.rgb_project, self.ir_project = projection(), projection()
            self.gate = nn.Linear(128, 64 if method == "vector_gate" else 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.head = regression_head(64)
        else:
            raise ValueError(method)

    def fuse(self, features):
        if self.method == "concat":
            return features
        rgb, ir = features.split(self.modality_features, dim=-1)
        rgb, ir = self.rgb_project(rgb), self.ir_project(ir)
        weight = torch.sigmoid(self.gate(torch.cat((rgb, ir), dim=-1)))
        return weight * rgb + (1 - weight) * ir

    def forward(self, features):
        return self.head(self.fuse(features))


def build_model(architecture):
    config.branches_for(architecture)
    return ModalityFusion(1280 if architecture.startswith("dino_farl_") else 768,
                          config.method_for(architecture))
