"""Single-frame RGB models for Exp2."""

import torch
import torch.nn as nn

from .config import (
    CLASSIFIER_HIDDEN,
    DROPOUT,
    FACE_EMBED_DIM,
    STAGE_BLOCKS,
    STAGE_CHANNELS,
    STEM_CHANNELS,
)


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)


class SingleFrameRGBNet(nn.Module):
    """One task-specific classifier for one RGB face frame [B, 3, H, W]."""

    def __init__(self, embed_dim=FACE_EMBED_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, STEM_CHANNELS, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(STEM_CHANNELS), STEM_CHANNELS),
            nn.SiLU(inplace=True),
        )
        blocks = []
        in_channels = STEM_CHANNELS
        for stage_index, (out_channels, n_blocks) in enumerate(zip(STAGE_CHANNELS, STAGE_BLOCKS)):
            for block_index in range(n_blocks):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(DROPOUT),
            nn.Linear(embed_dim, CLASSIFIER_HIDDEN),
            nn.SiLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

    def forward(self, face):
        features = self.projector(self.pool(self.encoder(self.stem(face))))
        return self.head(features)


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
