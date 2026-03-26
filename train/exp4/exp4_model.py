"""Experiment 04 models: lightweight/full 1D autoencoders for rPPG quality modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE1D(nn.Module):
    """1D denoising autoencoder with explicit latent bottleneck."""

    def __init__(
        self,
        channels,
        latent_channels,
        latent_dropout=0.1,
        io_downsample_factor=2,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.io_downsample_factor = max(1, int(io_downsample_factor))

        self.encoder = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c3, c4, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(c4, latent_channels, kernel_size=1),
            nn.Tanh(),
            nn.Dropout(latent_dropout),
            nn.Conv1d(latent_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(c4, c3, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c3, c2, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c2, c1, kernel_size=8, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(c1, 1, kernel_size=8, stride=2, padding=3),
        )

    def forward(self, x):
        in_len = x.shape[-1]

        # Process fewer samples for the same temporal span to expand effective RF.
        if self.io_downsample_factor > 1:
            x_proc = F.avg_pool1d(
                x,
                kernel_size=self.io_downsample_factor,
                stride=self.io_downsample_factor,
                ceil_mode=False,
            )
        else:
            x_proc = x

        z = self.encoder(x_proc)
        z = self.bottleneck(z)
        out_proc = self.decoder(z)
        if out_proc.shape[-1] != x_proc.shape[-1]:
            out_proc = F.interpolate(out_proc, size=x_proc.shape[-1], mode="linear", align_corners=False)

        if out_proc.shape[-1] != in_len:
            out = F.interpolate(out_proc, size=in_len, mode="linear", align_corners=False)
        else:
            out = out_proc
        return out


def build_exp4_model(variant="light"):
    variant = variant.lower()
    if variant == "light":
        return ConvAE1D(
            (8, 12, 16, 24),
            latent_channels=8,
            latent_dropout=0.1,
            io_downsample_factor=2,
        )
    if variant == "full":
        # Full model uses stronger temporal compression to increase effective RF.
        return ConvAE1D(
            (16, 24, 32, 48),
            latent_channels=12,
            latent_dropout=0.15,
            io_downsample_factor=4,
        )
    raise ValueError("variant must be 'light' or 'full'")
