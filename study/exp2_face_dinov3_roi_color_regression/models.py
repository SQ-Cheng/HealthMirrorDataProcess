"""Train-only color normalization followed by a compact fusion head."""

import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(self, color_mean=None, color_scale=None):
        super().__init__()
        self.register_buffer("color_mean", torch.zeros(164) if color_mean is None else torch.as_tensor(color_mean, dtype=torch.float32))
        self.register_buffer("color_scale", torch.ones(164) if color_scale is None else torch.as_tensor(color_scale, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(548, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(.25), nn.Linear(32, 1))

    def forward(self, features):
        color = (features[..., 384:] - self.color_mean) / self.color_scale
        return self.head(torch.cat((features[..., :384], color), dim=-1)).squeeze(-1)
