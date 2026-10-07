"""Two feature MLPs and a 97k-parameter spatial CNN, all trained from scratch."""

import torch
from torch import nn

from .config import DROPOUT, HISTOGRAM_FEATURES, STATISTICS_FEATURES


class ColorMLP(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(features))
        self.register_buffer("feature_std", torch.ones(features))
        self.network = nn.Sequential(
            nn.Linear(features, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.network((features - self.feature_mean) / self.feature_std)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 3
        for outgoing in (16, 32, 64, 120):
            layers.extend([nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, outgoing), nn.SiLU()])
            incoming = outgoing
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(120, 32), nn.LayerNorm(32), nn.SiLU(),
                                  nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, images):
        return self.head(self.encoder(images))


def build_model(architecture):
    if architecture == "color_histogram_mlp":
        return ColorMLP(HISTOGRAM_FEATURES)
    if architecture == "color_statistics_mlp":
        return ColorMLP(STATISTICS_FEATURES)
    if architecture == "small_cnn":
        return SmallCNN()
    raise ValueError(architecture)
