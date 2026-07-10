"""Face-only models for Exp2."""

import torch
import torch.nn as nn

from .config import CLASSIFIER_HIDDEN, DROPOUT, FACE_CHANNELS, FACE_EMBED_DIM


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class FaceOnlyCNN(nn.Module):
    """Compact CNN for one grayscale face frame (B, 1, 32, 32)."""

    def __init__(self, channels=FACE_CHANNELS, embed_dim=FACE_EMBED_DIM,
                 hidden=CLASSIFIER_HIDDEN, dropout=DROPOUT):
        super().__init__()
        blocks = []
        for idx in range(len(channels) - 1):
            blocks.append(ConvBlock(channels[idx], channels[idx + 1], dropout=0.08 + 0.04 * idx))
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, face):
        x = self.encoder(face)
        x = self.pool(x)
        x = self.projector(x)
        return self.head(x)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
