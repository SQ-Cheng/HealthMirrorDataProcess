"""Residual 3D CNN described in Biosensors 2025, 15, 485."""

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )

    def forward(self, inputs):
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class PaperResidual3DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock3D(64, 64),
            ResidualBlock3D(64, 64),
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 128),
            ResidualBlock3D(128, 256, stride=2),
            ResidualBlock3D(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output = nn.Linear(256, 1)
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, video):
        features = self.blocks(self.stem(video))
        return self.output(self.pool(features).flatten(1)).squeeze(1)


def parameter_count(model=None):
    model = model or PaperResidual3DRegressor()
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["PaperResidual3DRegressor", "ResidualBlock3D", "parameter_count"]
