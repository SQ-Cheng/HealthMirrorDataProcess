"""Shared Transformer model for single-signal masked reconstruction (Exp3 ECG/rPPG split)."""

import torch
import torch.nn as nn


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


class TemporalSelfAttention(nn.Module):
    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        y = x.transpose(1, 2)
        y_norm = self.norm(y)
        y_attn, _ = self.attn(y_norm, y_norm, y_norm, need_weights=False)
        y = y + y_attn
        y = y + self.ffn(self.norm(y))
        return y.transpose(1, 2)


class SingleReconTransformerLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Sequential(
            ConvNormAct(2, 48, kernel=7, stride=1, dropout=0.05),
            ConvNormAct(48, 64, kernel=5, stride=1, dropout=0.05),
        )
        self.attn1 = TemporalSelfAttention(64, heads=4, dropout=0.08)
        self.attn2 = TemporalSelfAttention(64, heads=4, dropout=0.08)
        self.out = nn.Sequential(
            ConvNormAct(64, 32, kernel=5, stride=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h = self.in_proj(x)
        h = self.attn1(h)
        h = self.attn2(h)
        return self.out(h)


class SingleReconTransformerFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Sequential(
            ConvNormAct(2, 64, kernel=9, stride=1, dropout=0.08),
            ConvNormAct(64, 80, kernel=7, stride=1, dropout=0.08),
        )
        self.attn = nn.Sequential(
            TemporalSelfAttention(80, heads=8, dropout=0.1),
            TemporalSelfAttention(80, heads=8, dropout=0.1),
            TemporalSelfAttention(80, heads=8, dropout=0.1),
            TemporalSelfAttention(80, heads=8, dropout=0.1),
        )
        self.out = nn.Sequential(
            ConvNormAct(80, 48, kernel=5, stride=1, dropout=0.06),
            ConvNormAct(48, 32, kernel=5, stride=1, dropout=0.06),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h = self.in_proj(x)
        h = self.attn(h)
        return self.out(h)


def build_single_recon_transformer_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return SingleReconTransformerLight()
    if variant == "full":
        return SingleReconTransformerFull()
    raise ValueError("variant must be 'light' or 'full'")
