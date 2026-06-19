"""Joint ECG+rPPG masked reconstruction model.

Takes 2-channel input (ECG + rPPG) and visible mask, outputs 2-channel reconstruction.
Supports 'light' and 'full' variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → Dropout."""

    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two ConvBlocks with ReLUskip connection."""

    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel=5, dilation=dilation, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, kernel=5, dilation=dilation, dropout=dropout)

    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + x)


# ──────────────────────────────────────────────
# Light variant
# ──────────────────────────────────────────────

class MaskedReconNetLight(nn.Module):
    """Compact masked-reconstruction model for joint ECG+rPPG."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(4, 24, kernel=9, stride=2, dropout=0.05),
            ConvBlock(24, 32, kernel=7, stride=2, dropout=0.05),
            ResidualBlock(32, dilation=1, dropout=0.05),
            ResidualBlock(32, dilation=2, dropout=0.05),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 24, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(24, 16, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 2, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        z = self.encoder(inp)
        out = self.decoder(z)
        if out.shape[-1] != x_masked.shape[-1]:
            out = F.interpolate(out, size=x_masked.shape[-1], mode="linear", align_corners=False)
        return out


# ──────────────────────────────────────────────
# Full variant
# ──────────────────────────────────────────────

class MaskedReconNetFull(nn.Module):
    """Deeper dilated masked-reconstruction model for joint ECG+rPPG."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(4, 32, kernel=11, stride=2, dropout=0.1),
            ConvBlock(32, 48, kernel=9, stride=2, dropout=0.1),
            ConvBlock(48, 64, kernel=7, stride=2, dropout=0.1),
        )
        self.body = nn.Sequential(
            ResidualBlock(64, dilation=1, dropout=0.1),
            ResidualBlock(64, dilation=2, dropout=0.1),
            ResidualBlock(64, dilation=4, dropout=0.1),
            ResidualBlock(64, dilation=8, dropout=0.1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 48, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(48, 32, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 24, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(24, 2, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        z = self.stem(inp)
        z = self.body(z)
        out = self.decoder(z)
        if out.shape[-1] != x_masked.shape[-1]:
            out = F.interpolate(out, size=x_masked.shape[-1], mode="linear", align_corners=False)
        return out


def build_joint_model(variant="light"):
    """Build a joint ECG+rPPG masked reconstruction model.

    Args:
        variant: 'light' or 'full'.
    """
    variant = variant.lower()
    if variant == "light":
        return MaskedReconNetLight()
    if variant == "full":
        return MaskedReconNetFull()
    raise ValueError(f"Unknown variant '{variant}'. Choose 'light' or 'full'.")
