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
    """U-Net-like lightweight model for one-channel reconstruction."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 24, kernel=9, stride=1, dropout=0.05)
        self.enc2 = ConvBlock(24, 32, kernel=7, stride=2, dropout=0.05)
        self.enc3 = ConvBlock(32, 48, kernel=7, stride=2, dropout=0.05)

        self.body = nn.Sequential(
            ResidualBlock(48, dilation=1, dropout=0.05),
            ResidualBlock(48, dilation=2, dropout=0.05),
        )

        self.up2 = nn.ConvTranspose1d(48, 32, kernel_size=8, stride=2, padding=3)
        self.dec2 = ConvBlock(32 + 32, 32, kernel=7, stride=1, dropout=0.05)
        self.up1 = nn.ConvTranspose1d(32, 24, kernel_size=8, stride=2, padding=3)
        self.dec1 = ConvBlock(24 + 24, 24, kernel=9, stride=1, dropout=0.05)

        self.refine = ConvBlock(24 + 2, 16, kernel=5, stride=1, dropout=0.02)
        self.out_conv = nn.Conv1d(16, 1, kernel_size=5, padding=2)

    @staticmethod
    def _resize_to(x, ref):
        if x.shape[-1] != ref.shape[-1]:
            x = F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)
        return x

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.body(e3)

        u2 = self._resize_to(self.up2(b), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self._resize_to(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        d1 = self._resize_to(d1, inp)
        f = self.refine(torch.cat([d1, inp], dim=1))
        out = self.out_conv(f)
        return out


class SingleReconNetFull(nn.Module):
    """U-Net-like moderate-capacity model with skip connections."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 32, kernel=11, stride=1, dropout=0.08)
        self.enc2 = ConvBlock(32, 48, kernel=9, stride=2, dropout=0.08)
        self.enc3 = ConvBlock(48, 64, kernel=7, stride=2, dropout=0.08)
        self.enc4 = ConvBlock(64, 80, kernel=7, stride=2, dropout=0.08)

        self.body = nn.Sequential(
            ResidualBlock(80, dilation=1, dropout=0.08),
            ResidualBlock(80, dilation=2, dropout=0.08),
            ResidualBlock(80, dilation=4, dropout=0.08),
        )

        self.up3 = nn.ConvTranspose1d(80, 64, kernel_size=8, stride=2, padding=3)
        self.dec3 = ConvBlock(64 + 64, 64, kernel=7, stride=1, dropout=0.08)
        self.up2 = nn.ConvTranspose1d(64, 48, kernel_size=8, stride=2, padding=3)
        self.dec2 = ConvBlock(48 + 48, 48, kernel=9, stride=1, dropout=0.08)
        self.up1 = nn.ConvTranspose1d(48, 32, kernel_size=8, stride=2, padding=3)
        self.dec1 = ConvBlock(32 + 32, 32, kernel=11, stride=1, dropout=0.08)

        self.refine = ConvBlock(32 + 2, 24, kernel=7, stride=1, dropout=0.04)
        self.out_conv = nn.Conv1d(24, 1, kernel_size=5, padding=2)

    @staticmethod
    def _resize_to(x, ref):
        if x.shape[-1] != ref.shape[-1]:
            x = F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)
        return x

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.body(e4)

        u3 = self._resize_to(self.up3(b), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self._resize_to(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self._resize_to(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        d1 = self._resize_to(d1, inp)
        f = self.refine(torch.cat([d1, inp], dim=1))
        out = self.out_conv(f)
        return out


def build_single_recon_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return SingleReconNetLight()
    if variant == "full":
        return SingleReconNetFull()
    raise ValueError("variant must be 'light' or 'full'")
