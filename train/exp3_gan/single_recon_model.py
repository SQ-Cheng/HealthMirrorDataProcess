"""TCN-GAN model components for single-signal masked reconstruction (Exp3 ECG/rPPG GAN split)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
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
    def __init__(self, channels, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation, bias=False)
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


class TCNGeneratorLight(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 48
        self.stem = ConvBlock(2, ch, kernel=7, dilation=1, dropout=0.05)
        self.tcn = nn.Sequential(
            DilatedResBlock(ch, kernel=3, dilation=1, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=4, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=8, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=2, dropout=0.05),
            DilatedResBlock(ch, kernel=3, dilation=1, dropout=0.05),
        )
        self.head = nn.Sequential(
            ConvBlock(ch + 2, 32, kernel=5, dilation=1, dropout=0.03),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x_masked, visible_mask):
        inp = torch.cat([x_masked, visible_mask], dim=1)
        z = self.stem(inp)
        z = self.tcn(z)
        return self.head(torch.cat([z, inp], dim=1))


class TCNGeneratorFull(nn.Module):
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


class SignalDiscriminator1D(nn.Module):
    """Patch-style 1D discriminator over reconstructed/real target signals."""

    def __init__(self, in_channels=1, base_channels=32):
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


def build_exp3_gan_models(variant="light"):
    variant = variant.lower()
    if variant == "light":
        generator = TCNGeneratorLight()
        discriminator = SignalDiscriminator1D(base_channels=16)
    elif variant == "full":
        generator = TCNGeneratorFull()
        discriminator = SignalDiscriminator1D(base_channels=32)
    else:
        raise ValueError("variant must be 'light' or 'full'")
    return generator, discriminator


def init_weights(module):
    classname = module.__class__.__name__
    if "Conv" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
