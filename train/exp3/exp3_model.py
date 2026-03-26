"""Experiment 03 models for rPPG -> (HR, SpO2) regression."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * 3
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=7, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        return self.relu(out + x)


class VitalSignsNetLight(nn.Module):
    """Small TCN-like model for quick CPU validation."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(1, 16, kernel=9, stride=2, dropout=0.05),
            ConvBlock(16, 32, kernel=9, stride=2, dropout=0.05),
        )
        self.tcn = nn.Sequential(
            ResidualTemporalBlock(32, dilation=1, dropout=0.05),
            ResidualTemporalBlock(32, dilation=2, dropout=0.05),
            ResidualTemporalBlock(32, dilation=4, dropout=0.05),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tcn(x)
        return self.head(x)


class VitalSignsNetFull(nn.Module):
    """ResNet-style model with deeper temporal context for reliable training."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(1, 32, kernel=15, stride=2, dropout=0.1),
            ConvBlock(32, 64, kernel=11, stride=2, dropout=0.1),
            ConvBlock(64, 96, kernel=9, stride=2, dropout=0.1),
        )
        self.res = nn.Sequential(
            ResidualTemporalBlock(96, dilation=1, dropout=0.1),
            ResidualTemporalBlock(96, dilation=2, dropout=0.1),
            ResidualTemporalBlock(96, dilation=4, dropout=0.1),
            ResidualTemporalBlock(96, dilation=8, dropout=0.1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        return self.head(x)


def build_exp3_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return VitalSignsNetLight()
    if variant == "full":
        return VitalSignsNetFull()
    raise ValueError("variant must be 'light' or 'full'")
