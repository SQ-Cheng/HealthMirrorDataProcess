"""Experiment 03X candidate models for masked ECG+rPPG reconstruction."""

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
        # x: [B,C,T] -> [B,T,C]
        y = x.transpose(1, 2)
        y_norm = self.norm(y)
        y_attn, _ = self.attn(y_norm, y_norm, y_norm, need_weights=False)
        y = y + y_attn
        y = y + self.ffn(self.norm(y))
        return y.transpose(1, 2)


class Exp3XUNetGated(nn.Module):
    """Mask-aware U-Net with gated skips for detail recovery."""

    def __init__(self):
        super().__init__()
        in_ch = 4
        self.e1 = ConvNormAct(in_ch, 32, kernel=9, stride=2, dropout=0.05)
        self.e2 = ConvNormAct(32, 48, kernel=7, stride=2, dropout=0.05)
        self.e3 = ConvNormAct(48, 64, kernel=5, stride=2, dropout=0.05)

        self.mid = nn.Sequential(
            GatedResidual(64, dilation=1, dropout=0.08),
            GatedResidual(64, dilation=2, dropout=0.08),
            GatedResidual(64, dilation=4, dropout=0.08),
        )

        self.d3 = nn.ConvTranspose1d(64, 48, kernel_size=8, stride=2, padding=3)
        self.d2 = nn.ConvTranspose1d(48, 32, kernel_size=8, stride=2, padding=3)
        self.d1 = nn.ConvTranspose1d(32, 24, kernel_size=8, stride=2, padding=3)

        self.skip3_gate = nn.Sequential(nn.Conv1d(48, 48, 1), nn.Sigmoid())
        self.skip2_gate = nn.Sequential(nn.Conv1d(32, 32, 1), nn.Sigmoid())

        self.out = nn.Conv1d(24, 2, kernel_size=5, padding=2)

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        s1 = self.e1(x)
        s2 = self.e2(s1)
        z = self.e3(s2)

        z = self.mid(z)
        z = F.gelu(self.d3(z))
        z = z + self.skip3_gate(s2) * s2
        z = F.gelu(self.d2(z))
        z = z + self.skip2_gate(s1) * s1
        z = F.gelu(self.d1(z))

        out = self.out(z)
        if out.shape[-1] != x_masked.shape[-1]:
            out = F.interpolate(out, size=x_masked.shape[-1], mode="linear", align_corners=False)
        return out


class Exp3XDualHead(nn.Module):
    """Shared encoder with ECG-detail and rPPG-smooth dual heads."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            ConvNormAct(4, 32, kernel=9, stride=2, dropout=0.08),
            ConvNormAct(32, 48, kernel=7, stride=2, dropout=0.08),
            ConvNormAct(48, 64, kernel=5, stride=2, dropout=0.08),
            GatedResidual(64, dilation=1, dropout=0.08),
            GatedResidual(64, dilation=2, dropout=0.08),
            GatedResidual(64, dilation=4, dropout=0.08),
        )
        self.ecg_head = nn.Sequential(
            nn.ConvTranspose1d(64, 48, kernel_size=8, stride=2, padding=3),
            nn.GELU(),
            nn.ConvTranspose1d(48, 32, kernel_size=8, stride=2, padding=3),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )
        self.rppg_head = nn.Sequential(
            nn.ConvTranspose1d(64, 48, kernel_size=8, stride=2, padding=3),
            nn.GELU(),
            nn.ConvTranspose1d(48, 32, kernel_size=8, stride=2, padding=3),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        z = self.enc(x)
        ecg = self.ecg_head(z)
        rppg = self.rppg_head(z)
        out = torch.cat([ecg, rppg], dim=1)
        if out.shape[-1] != x_masked.shape[-1]:
            out = F.interpolate(out, size=x_masked.shape[-1], mode="linear", align_corners=False)
        return out


class Exp3XTCNSSM(nn.Module):
    """Dilated TCN with gated state-space-like residual mixing."""

    def __init__(self):
        super().__init__()
        self.inp = ConvNormAct(4, 64, kernel=7, stride=1, dropout=0.08)
        self.blocks = nn.ModuleList(
            [
                GatedResidual(64, dilation=1, dropout=0.08),
                GatedResidual(64, dilation=2, dropout=0.08),
                GatedResidual(64, dilation=4, dropout=0.08),
                GatedResidual(64, dilation=8, dropout=0.08),
                GatedResidual(64, dilation=16, dropout=0.08),
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=1),
        )
        self.out = nn.Sequential(
            ConvNormAct(64, 32, kernel=5, dropout=0.05),
            nn.Conv1d(32, 2, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        h = h + self.mix(h)
        return self.out(h)


class Exp3XCrossAttention(nn.Module):
    """Cross-channel attention reconstructor for ECG<->rPPG coupling."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Sequential(
            ConvNormAct(4, 48, kernel=7, stride=1, dropout=0.08),
            ConvNormAct(48, 64, kernel=5, stride=1, dropout=0.08),
        )
        self.attn1 = TemporalSelfAttention(64, heads=4, dropout=0.08)
        self.attn2 = TemporalSelfAttention(64, heads=4, dropout=0.08)
        self.refine = nn.Sequential(
            GatedResidual(64, dilation=1, dropout=0.08),
            GatedResidual(64, dilation=2, dropout=0.08),
        )
        self.out = nn.Sequential(
            ConvNormAct(64, 32, kernel=5, stride=1, dropout=0.05),
            nn.Conv1d(32, 2, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h = self.in_proj(x)
        h = self.attn1(h)
        h = self.attn2(h)
        h = self.refine(h)
        return self.out(h)


class MambaLikeBlock(nn.Module):
    """Lightweight real-valued Mamba-style token mixer for 1D signals."""

    def __init__(self, channels, kernel=5, expand=2, dropout=0.1):
        super().__init__()
        hidden = channels * expand
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=kernel // 2, groups=hidden)
        self.out_proj = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,C,T] -> [B,T,C]
        y = x.transpose(1, 2)
        y = self.norm(y)
        v, g = torch.chunk(self.in_proj(y), 2, dim=-1)
        v = v.transpose(1, 2)
        v = self.dwconv(v)
        v = v.transpose(1, 2)
        y = F.silu(v) * F.silu(g)
        y = self.drop(self.out_proj(y))
        return x + y.transpose(1, 2)


class Exp3XMamba(nn.Module):
    """Optimized Mamba-style encoder-decoder for masked ECG+rPPG reconstruction."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(4, 48, kernel=7, stride=1, dropout=0.08),
            ConvNormAct(48, 64, kernel=5, stride=1, dropout=0.08),
        )
        self.b1 = MambaLikeBlock(64, kernel=5, expand=2, dropout=0.1)
        self.b2 = MambaLikeBlock(64, kernel=7, expand=2, dropout=0.1)
        self.b3 = MambaLikeBlock(64, kernel=9, expand=2, dropout=0.1)
        self.b4 = MambaLikeBlock(64, kernel=11, expand=2, dropout=0.1)
        self.b5 = MambaLikeBlock(64, kernel=7, expand=2, dropout=0.1)
        self.b6 = MambaLikeBlock(64, kernel=5, expand=2, dropout=0.1)

        self.ecg_refine = nn.Sequential(
            GatedResidual(64, dilation=1, dropout=0.08),
            GatedResidual(64, dilation=2, dropout=0.08),
        )
        self.rppg_refine = nn.Sequential(
            GatedResidual(64, dilation=1, dropout=0.08),
            GatedResidual(64, dilation=4, dropout=0.08),
        )

        self.ecg_out = nn.Sequential(
            ConvNormAct(64, 32, kernel=5, stride=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )
        self.rppg_out = nn.Sequential(
            ConvNormAct(64, 32, kernel=7, stride=1, dropout=0.05),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        h0 = self.stem(x)
        
        # Mamba with dense inward skip connections
        h1 = self.b1(h0)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        h4 = self.b4(h3) + h2
        h5 = self.b5(h4) + h1
        h6 = self.b6(h5) + h0
        
        ecg_feat = self.ecg_refine(h6)
        rppg_feat = self.rppg_refine(h6)

        ecg = self.ecg_out(ecg_feat)
        rppg = self.rppg_out(rppg_feat)

        out = torch.cat([ecg, rppg], dim=1)
        if out.shape[-1] != x_masked.shape[-1]:
            out = F.interpolate(out, size=x_masked.shape[-1], mode="linear", align_corners=False)
        return out


def build_exp3x_model(model_name):
    name = model_name.lower()
    if name == "unet_gated":
        return Exp3XUNetGated()
    if name == "dual_head":
        return Exp3XDualHead()
    if name == "tcn_ssm":
        return Exp3XTCNSSM()
    if name == "cross_attention":
        return Exp3XCrossAttention()
    if name == "mamba":
        return Exp3XMamba()
    raise ValueError("model_name must be one of: unet_gated, dual_head, tcn_ssm, cross_attention, mamba")
