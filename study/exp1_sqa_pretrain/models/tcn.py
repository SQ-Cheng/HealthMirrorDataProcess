"""TCN models for single-signal masked reconstruction.

Two variants:
- tcn256: 8 dilated residual blocks, receptive field ≈ 188 (target_length=256).
- tcn512: 10 dilated residual blocks, receptive field ≈ 513 (target_length=512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Two conv layers with dilation, residual connection, and SiLU activation."""

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
# TCN for target_length=256
# ──────────────────────────────────────────────

class SingleReconTCN256(nn.Module):
    """TCN for 256-point signals. Diamond dilation [1,2,4,8,16,8,4,2].

    Receptive field ≈ 188, covers ~73% of 256-point window.
    """

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
# TCN for target_length=512
# ──────────────────────────────────────────────

class SingleReconTCN512(nn.Module):
    """TCN for 512-point signals. Diamond dilation [1,2,4,8,16,32,16,8,4,2].

    Receptive field ≈ 513 at bottleneck, covers full 512-point window.
    """

    def __init__(self):
        super().__init__()
        ch = 72
        self.stem = ConvBlock(2, ch, kernel=9, dilation=1, dropout=0.08)
        self.tcn = nn.Sequential(
            DilatedResBlock(ch, kernel=5, dilation=1, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=2, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=4, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=8, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=16, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=32, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=16, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=8, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=4, dropout=0.08),
            DilatedResBlock(ch, kernel=5, dilation=2, dropout=0.08),
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


def build_tcn_model(variant="tcn256"):
    """Build a TCN model.

    Args:
        variant: 'tcn256' or 'tcn512'.
    """
    variant = variant.lower()
    if variant == "tcn256":
        return SingleReconTCN256()
    if variant == "tcn512":
        return SingleReconTCN512()
    raise ValueError(f"Unknown variant '{variant}'. Choose 'tcn256' or 'tcn512'.")
