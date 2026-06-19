"""Transformer models for single-signal masked reconstruction.

Uses temporal self-attention on the time dimension.
Supports 'light' and 'full' variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """Conv1d → BatchNorm → SiLU → Dropout."""

    def __init__(self, in_ch, out_ch, kernel=5, stride=1, dilation=1, dropout=0.1):
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


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention over the time dimension.

    Input: (B, C, L) → reshape → (B, L, C) → attention → (B, C, L).
    """

    def __init__(self, channels, num_heads=4, dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(channels, channels * 3)
        self.out_proj = nn.Linear(channels, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        x_t = x.transpose(1, 2)  # (B, L, C)

        qkv = self.qkv(x_t)  # (B, L, 3C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape to (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = attn @ v  # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.drop(self.out_proj(out))

        return x + out.transpose(1, 2)  # (B, C, L)


# ──────────────────────────────────────────────
# Light variant
# ──────────────────────────────────────────────

class SingleReconTransformerLight(nn.Module):
    """Light transformer model: input projection → 2 self-attention layers → output."""

    def __init__(self):
        super().__init__()
        embed_dim = 64
        self.in_proj = ConvNormAct(2, embed_dim, kernel=7, stride=1, dropout=0.1)
        self.attn1 = TemporalSelfAttention(embed_dim, num_heads=4, dropout=0.1)
        self.attn2 = TemporalSelfAttention(embed_dim, num_heads=4, dropout=0.1)
        self.out = nn.Sequential(
            ConvNormAct(embed_dim, 32, kernel=5, stride=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h = self.in_proj(x)
        h = self.attn1(h)
        h = self.attn2(h)
        return self.out(h)


# ──────────────────────────────────────────────
# Full variant
# ──────────────────────────────────────────────

class SingleReconTransformerFull(nn.Module):
    """Full transformer model: input projection → deeper self-attention → output."""

    def __init__(self):
        super().__init__()
        embed_dim = 96
        self.in_proj = ConvNormAct(2, embed_dim, kernel=9, stride=1, dropout=0.1)
        self.attn = nn.Sequential(
            TemporalSelfAttention(embed_dim, num_heads=4, dropout=0.1),
            TemporalSelfAttention(embed_dim, num_heads=4, dropout=0.1),
            TemporalSelfAttention(embed_dim, num_heads=4, dropout=0.1),
            TemporalSelfAttention(embed_dim, num_heads=4, dropout=0.1),
        )
        self.out = nn.Sequential(
            ConvNormAct(embed_dim, 48, kernel=5, stride=1, dropout=0.06),
            ConvNormAct(48, 32, kernel=5, stride=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h = self.in_proj(x)
        h = self.attn(h)
        return self.out(h)


def build_transformer_model(variant="light"):
    """Build a transformer model.

    Args:
        variant: 'light' or 'full'.
    """
    variant = variant.lower()
    if variant == "light":
        return SingleReconTransformerLight()
    if variant == "full":
        return SingleReconTransformerFull()
    raise ValueError(f"Unknown variant '{variant}'. Choose 'light' or 'full'.")
