"""TCN model for masked signal reconstruction.

Dilated Temporal Convolutional Network with diamond dilation stack.
Target ~450k-640k params (varies with target_length).
Works for target_length ∈ [256, 1024].

Architecture:
    Stem: Conv1d(k=9) → 64 channels
    TCN:  N× DilatedResBlock(k=5) in diamond dilation pattern
          (N computed to cover full target_length)
    Head: Concatenate input → ConvBlock → Conv1d → 1 channel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → SiLU → Dropout1d."""

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
    """Two Conv1d(k) with dilation + skip connection."""

    def __init__(self, channels, kernel=5, dilation=1, dropout=0.1):
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
# Dilation schedule
# ──────────────────────────────────────────────

def _compute_tcn_dilations(target_length, kernel_size=5):
    """Compute diamond dilation stack covering `target_length`.

    RF = 1 + (k-1)·Σ(dilations) ≥ target_length
    Diamond sum: 3·max_d − 2
    """
    needed_sum = (target_length - 1) / (kernel_size - 1)
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
# TCN Model
# ──────────────────────────────────────────────

class MaskedReconTCN(nn.Module):
    """Dilated TCN for masked signal reconstruction.

    Args:
        target_length: Input sequence length (e.g. 256, 512, 1024).
    """

    def __init__(self, target_length=256):
        super().__init__()
        ch = 64

        self.stem = ConvBlock(2, ch, kernel=9, dilation=1, dropout=0.08)

        dilations = _compute_tcn_dilations(target_length, kernel_size=5)
        self.tcn = nn.Sequential(*[
            DilatedResBlock(ch, kernel=5, dilation=d, dropout=0.1) for d in dilations
        ])

        self.head = nn.Sequential(
            ConvBlock(ch + 2, 40, kernel=7, dilation=1, dropout=0.05),
            nn.Conv1d(40, 1, kernel_size=5, padding=2),
        )

        self._target_length = target_length

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)      # (B, 2, L)
        z = self.stem(inp)                                     # (B, 64, L)
        z = self.tcn(z)                                        # (B, 64, L)
        return self.head(torch.cat([z, inp], dim=1))           # (B, 1, L)


def build_tcn_model(target_length=256):
    """Build the TCN model.

    Args:
        target_length: Input sequence length.
    """
    return MaskedReconTCN(target_length=target_length)
