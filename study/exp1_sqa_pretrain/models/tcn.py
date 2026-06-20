"""Separable TCN encoder and convolutional decoder for masked reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d followed by batch normalization, SiLU, and dropout."""

    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False
            ),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout1d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DilatedResBlock(nn.Module):
    """Two dilated convolutions with a residual connection."""

    def __init__(self, channels, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x):
        residual = self.drop(F.silu(self.bn1(self.conv1(x))))
        residual = self.drop(self.bn2(self.conv2(residual)))
        return F.silu(x + residual)


def _compute_tcn_dilations(target_length, kernel_size=5):
    """Return a diamond dilation schedule covering the input sequence."""
    needed_sum = (target_length - 1) / (kernel_size - 1)
    max_dilation = 1
    while 3 * max_dilation - 2 < needed_sum:
        max_dilation *= 2

    dilations = []
    dilation = 1
    while dilation <= max_dilation:
        dilations.append(dilation)
        dilation *= 2
    return dilations + dilations[-2::-1]


class TCNEncoder(nn.Module):
    """Full-resolution temporal convolutional encoder."""

    out_channels = 64

    def __init__(self, target_length=256):
        super().__init__()
        self.stem = ConvBlock(2, self.out_channels, kernel=9, dropout=0.08)
        self.tcn = nn.Sequential(*[
            DilatedResBlock(
                self.out_channels, kernel=5, dilation=dilation, dropout=0.1
            )
            for dilation in _compute_tcn_dilations(target_length)
        ])

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        return self.tcn(self.stem(x))


class TCNConvDecoder(nn.Module):
    """Convolutional decoder operating only on TCN latent features."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBlock(64, 40, kernel=7, dropout=0.05),
            nn.Conv1d(40, 1, kernel_size=5, padding=2),
        )

    def forward(self, latent):
        return self.decoder(latent)


class MaskedReconTCN(nn.Module):
    """Composition of independently usable TCN encoder and decoder models."""

    def __init__(self, target_length=256):
        super().__init__()
        self.encoder = TCNEncoder(target_length=target_length)
        self.decoder = TCNConvDecoder()

    def forward(self, x_masked, visible_mask):
        return self.decoder(self.encoder(x_masked, visible_mask))


def build_tcn_encoder(target_length=256):
    return TCNEncoder(target_length=target_length)


def build_tcn_decoder():
    return TCNConvDecoder()


def build_tcn_model(target_length=256):
    return MaskedReconTCN(target_length=target_length)
