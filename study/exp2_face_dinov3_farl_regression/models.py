"""Direct concatenation and compact elementwise-gated expert fusion."""

import torch
from torch import nn


def regression_head(incoming):
    return nn.Sequential(nn.Linear(incoming, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(.25), nn.Linear(32, 1))


class ConcatHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = regression_head(896)

    def forward(self, features):
        return self.head(features)


class GatedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino_project = nn.Sequential(nn.Linear(384, 64), nn.LayerNorm(64), nn.SiLU())
        self.farl_project = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.SiLU())
        self.gate = nn.Linear(128, 64)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.head = regression_head(64)

    def forward(self, features):
        dino, farl = features.split((384, 512), dim=1)
        dino, farl = self.dino_project(dino), self.farl_project(farl)
        gate = self.gate(torch.cat((dino, farl), dim=1)).sigmoid()
        return self.head(gate * dino + (1 - gate) * farl)


def build_model(architecture):
    if architecture == "concat":
        return ConcatHead()
    if architecture == "gated":
        return GatedHead()
    raise ValueError(architecture)


class FrozenDualPredictor(nn.Module):
    """Online equivalent; input branches receive their respective normalization."""

    def __init__(self, dino, farl, variant):
        super().__init__()
        self.dino = dino.eval().requires_grad_(False)
        self.farl = farl.eval().requires_grad_(False)
        self.head = build_model(variant)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        self.farl.eval()
        return self

    def forward(self, dino_images, farl_images):
        with torch.no_grad():
            features = torch.cat((self.dino(dino_images), self.farl(farl_images)), dim=1)
        return self.head(features)
