"""Shared Mamba-style model for single-signal masked reconstruction (Exp3 ECG/rPPG split)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class GatedResidual(nn.Module):
    def __init__(self, channels, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=kernel, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels * 2),
        )
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.pre(x)
        a, g = torch.chunk(h, 2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(g)
        h = self.post(h)
        return x + h


class MambaLikeBlock(nn.Module):
    def __init__(self, channels, kernel=5, expand=2, dropout=0.1):
        super().__init__()
        hidden = channels * expand
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=kernel // 2, groups=hidden)
        self.out_proj = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.norm(y)
        v, g = torch.chunk(self.in_proj(y), 2, dim=-1)
        v = v.transpose(1, 2)
        v = self.dwconv(v)
        v = v.transpose(1, 2)
        y = F.silu(v) * F.silu(g)
        y = self.drop(self.out_proj(y))
        return x + y.transpose(1, 2)


class SingleReconMambaLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 48, kernel=7, stride=1, dropout=0.08),
            ConvNormAct(48, 64, kernel=5, stride=1, dropout=0.08),
        )
        self.b1 = MambaLikeBlock(64, kernel=5, expand=2, dropout=0.1)
        self.b2 = MambaLikeBlock(64, kernel=7, expand=2, dropout=0.1)
        self.b3 = MambaLikeBlock(64, kernel=9, expand=2, dropout=0.1)
        self.b4 = MambaLikeBlock(64, kernel=7, expand=2, dropout=0.1)
        self.out = nn.Sequential(
            ConvNormAct(64, 32, kernel=5, stride=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h0 = self.stem(x)
        h1 = self.b1(h0)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        h4 = self.b4(h3) + h1
        return self.out(h4)


class SingleReconMambaFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 64, kernel=9, stride=1, dropout=0.1),
            ConvNormAct(64, 80, kernel=7, stride=1, dropout=0.1),
        )
        self.b1 = MambaLikeBlock(80, kernel=5, expand=2, dropout=0.1)
        self.b2 = MambaLikeBlock(80, kernel=7, expand=2, dropout=0.1)
        self.b3 = MambaLikeBlock(80, kernel=9, expand=2, dropout=0.1)
        self.b4 = MambaLikeBlock(80, kernel=11, expand=2, dropout=0.1)
        self.b5 = MambaLikeBlock(80, kernel=9, expand=2, dropout=0.1)
        self.b6 = MambaLikeBlock(80, kernel=7, expand=2, dropout=0.1)
        self.refine = nn.Sequential(
            GatedResidual(80, dilation=1, dropout=0.08),
            GatedResidual(80, dilation=2, dropout=0.08),
        )
        self.out = nn.Sequential(
            ConvNormAct(80, 48, kernel=5, stride=1, dropout=0.06),
            ConvNormAct(48, 32, kernel=5, stride=1, dropout=0.06),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h0 = self.stem(x)
        h1 = self.b1(h0)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        h4 = self.b4(h3) + h2
        h5 = self.b5(h4) + h1
        h6 = self.b6(h5) + h0
        h = self.refine(h6)
        return self.out(h)


def build_single_recon_mamba_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return SingleReconMambaLight()
    if variant == "full":
        return SingleReconMambaFull()
    raise ValueError("variant must be 'light' or 'full'")
