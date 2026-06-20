"""1D ResNet encoder and convolutional decoder for masked reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d followed by batch normalization, SiLU, and dropout."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch, out_ch, kernel, stride=stride, padding=padding,
                dilation=dilation, bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout1d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResNetBlock1D(nn.Module):
    """Two-layer residual block with an optional projection shortcut."""

    def __init__(
        self, in_ch, out_ch, kernel=5, stride=1, dilation=1,
        second_kernel=3, dropout=0.1,
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            in_ch, out_ch, kernel, stride=stride, dilation=dilation, dropout=dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_ch, out_ch, second_kernel,
                padding=(second_kernel // 2) * dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm1d(out_ch),
            nn.Dropout1d(dropout),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        return F.silu(self.conv2(self.conv1(x)) + self.shortcut(x))


def _compute_dilations(bottleneck_length, kernel_size=5):
    """Return a diamond dilation schedule covering the bottleneck sequence."""
    needed_sum = (bottleneck_length - 1) / (kernel_size - 1)
    max_dilation = 1
    while 3 * max_dilation - 2 < needed_sum:
        max_dilation *= 2

    dilations = []
    dilation = 1
    while dilation <= max_dilation:
        dilations.append(dilation)
        dilation *= 2
    return dilations + dilations[-2::-1]


class ResNetEncoder1D(nn.Module):
    """Residual encoder producing a single, downsampled latent tensor."""

    out_channels = 64
    downsample_factor = 8

    def __init__(self, target_length=256):
        super().__init__()
        self.stem = ConvBlock(2, 28, kernel=11, dropout=0.08)
        self.stages = nn.Sequential(
            ResNetBlock1D(28, 40, kernel=9, stride=2, dropout=0.08),
            ResNetBlock1D(40, 52, kernel=7, stride=2, dropout=0.08),
            ResNetBlock1D(52, 64, kernel=7, stride=2, dropout=0.08),
        )
        bottleneck_length = max(1, target_length // self.downsample_factor)
        self.bottleneck = nn.Sequential(*[
            ResNetBlock1D(
                64, 64, kernel=5, second_kernel=5,
                dilation=dilation, dropout=0.1,
            )
            for dilation in _compute_dilations(bottleneck_length)
        ])

    def forward(self, x_masked, visible_mask):
        x = torch.cat([x_masked, visible_mask], dim=1)
        return self.bottleneck(self.stages(self.stem(x)))


class ConvDecoder1D(nn.Module):
    """Decode a ResNet latent tensor without encoder skip connections."""

    def __init__(self, target_length=256):
        super().__init__()
        self.target_length = target_length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 52, kernel_size=8, stride=2, padding=3),
            ResNetBlock1D(52, 52, kernel=7, second_kernel=7, dropout=0.08),
            nn.ConvTranspose1d(52, 40, kernel_size=8, stride=2, padding=3),
            ResNetBlock1D(40, 40, kernel=9, second_kernel=9, dropout=0.08),
            nn.ConvTranspose1d(40, 28, kernel_size=8, stride=2, padding=3),
            ResNetBlock1D(28, 28, kernel=11, second_kernel=11, dropout=0.08),
            ConvBlock(28, 20, kernel=7, dropout=0.04),
            nn.Conv1d(20, 1, kernel_size=5, padding=2),
        )

    def forward(self, latent, output_length=None):
        output = self.decoder(latent)
        output_length = output_length or self.target_length
        if output.shape[-1] != output_length:
            output = F.interpolate(
                output, size=output_length, mode="linear", align_corners=False
            )
        return output


class MaskedReconResNet(nn.Module):
    """Composition of independently usable ResNet encoder and decoder models."""

    def __init__(self, target_length=256):
        super().__init__()
        self.encoder = ResNetEncoder1D(target_length=target_length)
        self.decoder = ConvDecoder1D(target_length=target_length)

    def forward(self, x_masked, visible_mask):
        latent = self.encoder(x_masked, visible_mask)
        return self.decoder(latent, output_length=x_masked.shape[-1])


def build_resnet_encoder(target_length=256):
    return ResNetEncoder1D(target_length=target_length)


def build_resnet_decoder(target_length=256):
    return ConvDecoder1D(target_length=target_length)


def build_resnet_model(target_length=256):
    return MaskedReconResNet(target_length=target_length)
