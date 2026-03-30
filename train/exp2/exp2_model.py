"""
Experiment 02 Model
===================
Bidirectional GAN components for ECG <-> rPPG translation.
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Simple residual block that preserves temporal length."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator1D(nn.Module):
    """1D encoder-residual-decoder generator."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, n_res_blocks=4):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, padding=7, bias=False),
            nn.InstanceNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.InstanceNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=9, stride=2, padding=4, bias=False),
            nn.InstanceNorm1d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(base_channels * 4) for _ in range(n_res_blocks)]
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose1d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=8,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.InstanceNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(
                base_channels * 2,
                base_channels,
                kernel_size=8,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.InstanceNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels, out_channels, kernel_size=15, padding=7),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.res_blocks(x)
        x = self.dec(x)
        return x


class Discriminator1D(nn.Module):
    """Patch-style 1D discriminator."""

    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.InstanceNorm1d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=9, stride=2, padding=4, bias=False),
            nn.InstanceNorm1d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels * 8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def build_exp2_models(variant="full"):
    """Build lightweight or full GAN components for Experiment 2."""
    variant = variant.lower()
    if variant == "light":
        g_e2r = Generator1D(base_channels=16, n_res_blocks=2)
        g_r2e = Generator1D(base_channels=16, n_res_blocks=2)
        d_rppg = Discriminator1D(base_channels=16)
        d_ecg = Discriminator1D(base_channels=16)
    elif variant == "full":
        g_e2r = Generator1D(base_channels=64, n_res_blocks=6)
        g_r2e = Generator1D(base_channels=64, n_res_blocks=6)
        d_rppg = Discriminator1D(base_channels=64)
        d_ecg = Discriminator1D(base_channels=64)
    else:
        raise ValueError("variant must be 'light' or 'full'")

    return g_e2r, g_r2e, d_rppg, d_ecg


def init_weights(module):
    """Normal initialization commonly used for GAN training stability."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
