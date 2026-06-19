"""GAN model for single-signal masked reconstruction.

Generator: TCN-based architecture (same as TCN generator).
Discriminator: PatchGAN-style 1D discriminator.
Supports 'light' and 'full' variants for the generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → SiLU → Dropout."""

    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout1d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DilatedResBlock(nn.Module):
    """Two conv layers with dilation and residual connection."""

    def __init__(self, channels, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.silu(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.drop(h)
        return F.silu(x + h)


# ──────────────────────────────────────────────
# Generators
# ──────────────────────────────────────────────

class TCNGeneratorLight(nn.Module):
    """Light TCN generator with smaller channel count."""

    def __init__(self):
        super().__init__()
        ch = 48
        self.stem = ConvBlock(2, ch, kernel=9, dilation=1, dropout=0.08)
        self.tcn = nn.Sequential(
            DilatedResBlock(ch, kernel=3, dilation=1, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=4, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=8, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=4, dropout=0.08),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.08),
        )
        self.head = nn.Sequential(
            ConvBlock(ch + 2, 32, kernel=7, dilation=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        z = self.stem(inp)
        z = self.tcn(z)
        return self.head(torch.cat([z, inp], dim=1))


class TCNGeneratorFull(nn.Module):
    """Full TCN generator with larger channel count and deeper TCN."""

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
        return self.head(torch.cat([z, inp], dim=1))


# ──────────────────────────────────────────────
# Discriminator (PatchGAN-style)
# ──────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """1D PatchGAN discriminator.

    Takes masked input and reconstruction (or target), outputs patch-level
    real/fake scores.
    """

    def __init__(self, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 1, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x, masked):
        """x: predicted or target signal (B,1,L), masked: masked input (B,1,L)."""
        inp = torch.cat([x, masked], dim=1)
        return self.net(inp)


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_gan_generator(variant="light"):
    """Build a GAN generator.

    Args:
        variant: 'light' or 'full'.
    """
    variant = variant.lower()
    if variant == "light":
        return TCNGeneratorLight()
    if variant == "full":
        return TCNGeneratorFull()
    raise ValueError(f"Unknown variant '{variant}'. Choose 'light' or 'full'.")


def build_gan_discriminator():
    """Build a PatchGAN discriminator."""
    return PatchDiscriminator()
