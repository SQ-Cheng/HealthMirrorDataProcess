"""CNN model for masked signal reconstruction.

UNet-style encoder-decoder with skip connections and dilated residual bottleneck.
Target ~560k parameters. Works for target_length ∈ [256, 1024].

Architecture:
    Encoder: 3× stride-2 ConvBlocks (8× spatial reduction)
    Bottleneck: N dilated ResidualBlocks (N computed to cover target_length/8)
    Decoder: 3× ConvTranspose + skip connections + ConvBlock refine
    Output: Conv1d → 1 channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → SiLU → Dropout."""

    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad,
                      dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two ConvBlocks with skip connection."""

    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel=5, dilation=dilation, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, kernel=5, dilation=dilation, dropout=dropout)

    def forward(self, x):
        return F.silu(self.conv2(self.conv1(x)) + x)


# ──────────────────────────────────────────────
# Dilation schedule
# ──────────────────────────────────────────────

def _compute_bottleneck_dilations(bottleneck_len, kernel_size=5):
    """Compute diamond dilation stack so receptive field covers bottleneck_len.

    RF = 1 + (k-1) * sum(dilations) ≥ bottleneck_len
    Diamond stack: [1, 2, 4, …, max_d, …, 4, 2, 1]  →  sum = 3·max_d - 2
    """
    needed_sum = (bottleneck_len - 1) / (kernel_size - 1)
    max_d = 1
    while 3 * max_d - 2 < needed_sum:
        max_d *= 2
    dilations = []
    d = 1
    while d <= max_d:
        dilations.append(d)
        d *= 2
    dilations += dilations[-2::-1]
    return dilations


# ──────────────────────────────────────────────
# CNN Model
# ──────────────────────────────────────────────

class MaskedReconCNN(nn.Module):
    """UNet CNN for masked signal reconstruction.

    Args:
        target_length: Input signal length (e.g. 256, 512, 1024).
    """

    def __init__(self, target_length=256):
        super().__init__()
        # ── Encoder ──────────────────────────
        self.enc1 = ConvBlock(2, 28, kernel=11, stride=1, dropout=0.08)
        self.enc2 = ConvBlock(28, 40, kernel=9, stride=2, dropout=0.08)
        self.enc3 = ConvBlock(40, 52, kernel=7, stride=2, dropout=0.08)
        self.enc4 = ConvBlock(52, 64, kernel=7, stride=2, dropout=0.08)

        # ── Bottleneck ───────────────────────
        bottleneck_len = target_length // 8  # after 3× stride-2
        dilations = _compute_bottleneck_dilations(bottleneck_len)
        self.bottleneck = nn.Sequential(*[
            ResidualBlock(64, dilation=d, dropout=0.1) for d in dilations
        ])

        # ── Decoder (with skip connections) ──
        self.up3 = nn.ConvTranspose1d(64, 52, kernel_size=8, stride=2, padding=3)
        self.dec3 = ConvBlock(52 + 52, 52, kernel=7, stride=1, dropout=0.08)

        self.up2 = nn.ConvTranspose1d(52, 40, kernel_size=8, stride=2, padding=3)
        self.dec2 = ConvBlock(40 + 40, 40, kernel=9, stride=1, dropout=0.08)

        self.up1 = nn.ConvTranspose1d(40, 28, kernel_size=8, stride=2, padding=3)
        self.dec1 = ConvBlock(28 + 28, 28, kernel=11, stride=1, dropout=0.08)

        # ── Output head ──────────────────────
        self.refine = ConvBlock(28 + 2, 20, kernel=7, stride=1, dropout=0.04)
        self.out_conv = nn.Conv1d(20, 1, kernel_size=5, padding=2)

        self._target_length = target_length

    @staticmethod
    def _resize_to(x, ref):
        if x.shape[-1] != ref.shape[-1]:
            x = F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)
        return x

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)      # (B, 2, L)

        e1 = self.enc1(inp)                                    # (B, 28, L)
        e2 = self.enc2(e1)                                     # (B, 40, L/2)
        e3 = self.enc3(e2)                                     # (B, 52, L/4)
        e4 = self.enc4(e3)                                     # (B, 64, L/8)

        b = self.bottleneck(e4)                                # (B, 64, L/8)

        d3 = self.dec3(torch.cat([self._resize_to(self.up3(b), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._resize_to(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._resize_to(self.up1(d2), e1), e1], dim=1))

        f = self.refine(torch.cat([self._resize_to(d1, inp), inp], dim=1))
        return self.out_conv(f)


def build_cnn_model(target_length=256):
    """Build the CNN model.

    Args:
        target_length: Input sequence length.
    """
    return MaskedReconCNN(target_length=target_length)
