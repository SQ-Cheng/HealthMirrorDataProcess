"""Shared model for single-signal masked reconstruction (Exp3 ECG/rPPG split)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel=5, dilation=dilation, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, kernel=5, dilation=dilation, dropout=dropout)

    def forward(self, x):
        return F.silu(self.conv2(self.conv1(x)) + x)


class SingleReconNetLight(nn.Module):
    """Balanced lightweight model for one-channel reconstruction."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 24, kernel=9, stride=1, dropout=0.05)
        self.enc2 = ConvBlock(24, 32, kernel=7, stride=1, dropout=0.05)
        
        self.body = nn.Sequential(
            ResidualBlock(32, dilation=1, dropout=0.05),
            ResidualBlock(32, dilation=2, dropout=0.05),
        )
        
        self.dec2 = ConvBlock(32 + 24, 24, kernel=7, stride=1, dropout=0.05)
        self.dec1 = ConvBlock(24 + 2, 16, kernel=9, stride=1, dropout=0.05)
        
        self.out_conv = nn.Conv1d(16, 1, kernel_size=5, padding=2)

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        
        b = self.body(e2)
        
        d2 = self.dec2(torch.cat([b, e1], dim=1))
        d1 = self.dec1(torch.cat([d2, inp], dim=1))
        
        out = self.out_conv(d1)
        return out


class SingleReconNetFull(nn.Module):
    """Moderate-capacity model with dilated residual body."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 32, kernel=11, stride=1, dropout=0.08)
        self.enc2 = ConvBlock(32, 48, kernel=9, stride=1, dropout=0.08)
        self.enc3 = ConvBlock(48, 64, kernel=7, stride=1, dropout=0.08)
        
        self.body = nn.Sequential(
            ResidualBlock(64, dilation=1, dropout=0.08),
            ResidualBlock(64, dilation=2, dropout=0.08),
            ResidualBlock(64, dilation=4, dropout=0.08),
        )
        
        self.dec3 = ConvBlock(64 + 48, 48, kernel=7, stride=1, dropout=0.08)
        self.dec2 = ConvBlock(48 + 32, 32, kernel=9, stride=1, dropout=0.08)
        self.dec1 = ConvBlock(32 + 2, 24, kernel=11, stride=1, dropout=0.08)
        
        self.out_conv = nn.Conv1d(24, 1, kernel_size=5, padding=2)

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.body(e3)
        
        d3 = self.dec3(torch.cat([b, e2], dim=1))
        d2 = self.dec2(torch.cat([d3, e1], dim=1))
        d1 = self.dec1(torch.cat([d2, inp], dim=1))
        
        out = self.out_conv(d1)
        return out


def build_single_recon_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return SingleReconNetLight()
    if variant == "full":
        return SingleReconNetFull()
    raise ValueError("variant must be 'light' or 'full'")
