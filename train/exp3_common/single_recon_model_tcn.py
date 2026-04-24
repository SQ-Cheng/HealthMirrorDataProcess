"""Shared TCN model for single-signal masked reconstruction (Exp3 ECG/rPPG TCN split)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.silu(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.drop(h)
        return F.silu(x + h)


class SingleReconTCNLight(nn.Module):
    """Compact TCN model with dilated residual stack."""

    def __init__(self):
        super().__init__()
        ch = 48
        self.stem = ConvBlock(2, ch, kernel=7, dilation=1, dropout=0.05)
        self.tcn = nn.Sequential(
            DilatedResBlock(ch, kernel=3, dilation=1, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=4, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=8, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=1, dropout=0.05),
        )
        self.head = nn.Sequential(
            ConvBlock(ch + 2, 32, kernel=5, dilation=1, dropout=0.03),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        z = self.stem(inp)
        z = self.tcn(z)
        out = self.head(torch.cat([z, inp], dim=1))
        return out


class SingleReconTCNFull(nn.Module):
    """Moderate-capacity TCN model with larger receptive field."""

    def __init__(self):
        super().__init__()
        ch = 72
        self.stem = ConvBlock(2, ch, kernel=9, dilation=1, dropout=0.08)
        self.tcn = nn.Sequential(
            DilatedResBlock(ch, kernel=3, dilation=1, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=4, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=8, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=16, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=8, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=4, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.08),
        )
        self.head = nn.Sequential(
            ConvBlock(ch + 2, 40, kernel=7, dilation=1, dropout=0.05),
            nn.Conv1d(40, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        z = self.stem(inp)
        z = self.tcn(z)
        out = self.head(torch.cat([z, inp], dim=1))
        return out


def build_single_recon_tcn_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return SingleReconTCNLight()
    if variant == "full":
        return SingleReconTCNFull()
    raise ValueError("variant must be 'light' or 'full'")
