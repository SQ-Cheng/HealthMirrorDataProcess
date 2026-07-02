"""Deep learning model architectures for Exp2: multi-modal lab test prediction.

Architecture overview:
    ECGEncoder:   1D CNN with residual blocks → 128-d embedding
    FaceEncoder:  2D CNN → 128-d embedding
    M3TNet:       Multi-Modal Multi-Task Network
                  Fuses ECG + Face embeddings → shared MLP → 15 task-specific heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DROPOUT,
    ECG_EMBED_DIM,
    ECG_ENC_CHANNELS,
    ECG_ENC_KERNELS,
    ECG_ENC_STRIDES,
    FACE_EMBED_DIM,
    FACE_ENC_CHANNELS,
    FUSION_HIDDEN,
    TARGETS,
)


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════

class Conv1dBlock(nn.Module):
    """Conv1d → BatchNorm → SiLU → Dropout."""

    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Conv2dBlock(nn.Module):
    """Conv2d → BatchNorm → SiLU → Dropout."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Residual1dBlock(nn.Module):
    """Two Conv1dBlocks with residual connection and optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = Conv1dBlock(in_ch, out_ch, kernel_size=5, stride=stride,
                                 dropout=dropout)
        self.conv2 = Conv1dBlock(out_ch, out_ch, kernel_size=5, stride=1,
                                 dropout=dropout)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.silu(out + residual)


# ═══════════════════════════════════════════════════════════════════════
# ECG Encoder (1D CNN)
# ═══════════════════════════════════════════════════════════════════════

class ECGEncoder(nn.Module):
    """1D CNN encoder for raw ECG signals.

    Input:  (B, 1, ECG_LENGTH)   e.g. (B, 1, 256)
    Output: (B, ECG_EMBED_DIM)    e.g. (B, 128)
    """

    def __init__(self, in_channels=1, embed_dim=ECG_EMBED_DIM,
                 enc_channels=ECG_ENC_CHANNELS, enc_kernels=ECG_ENC_KERNELS,
                 enc_strides=ECG_ENC_STRIDES, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(enc_channels) - 1):
            layers.append(
                Residual1dBlock(
                    enc_channels[i], enc_channels[i + 1],
                    stride=enc_strides[i], dropout=dropout,
                )
            )
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(enc_channels[-1], embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C, L)
        x = self.encoder(x)          # (B, last_ch, L')
        x = self.pool(x).squeeze(-1) # (B, last_ch)
        x = self.proj(x)             # (B, embed_dim)
        return x


# ═══════════════════════════════════════════════════════════════════════
# Face Encoder (2D CNN)
# ═══════════════════════════════════════════════════════════════════════

class FaceEncoder(nn.Module):
    """2D CNN encoder for face/rPPG images.

    Input:  (B, 1, FACE_SIZE, FACE_SIZE)   e.g. (B, 1, 32, 32)
    Output: (B, FACE_EMBED_DIM)             e.g. (B, 128)
    """

    def __init__(self, in_channels=1, embed_dim=FACE_EMBED_DIM,
                 enc_channels=FACE_ENC_CHANNELS, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(enc_channels) - 1):
            layers.append(
                Conv2dBlock(enc_channels[i], enc_channels[i + 1],
                            kernel_size=3, stride=2, dropout=dropout)
            )
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(enc_channels[-1], embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.encoder(x)          # (B, last_ch, H', W')
        x = self.pool(x)             # (B, last_ch, 1, 1)
        x = x.view(x.size(0), -1)    # (B, last_ch)
        x = self.proj(x)             # (B, embed_dim)
        return x


# ═══════════════════════════════════════════════════════════════════════
# M3TNet: Multi-Modal Multi-Task Network
# ═══════════════════════════════════════════════════════════════════════

class M3TNet(nn.Module):
    """Multi-Modal Multi-Task Network for lab test prediction.

    Fuses ECG and Face embeddings through a shared MLP, then predicts
    15 binary lab-test abnormality targets via independent heads.

    Args:
        num_tasks: Number of binary classification tasks (default: len(TARGETS)=15).
        ecg_embed_dim: Dimension of ECG embedding.
        face_embed_dim: Dimension of Face embedding.
        fusion_hidden: List of hidden dimensions for fusion MLP.
        dropout: Dropout rate in fusion MLP.
    """

    def __init__(self, num_tasks=len(TARGETS), ecg_embed_dim=ECG_EMBED_DIM,
                 face_embed_dim=FACE_EMBED_DIM, fusion_hidden=FUSION_HIDDEN,
                 dropout=DROPOUT):
        super().__init__()
        self.ecg_encoder = ECGEncoder(embed_dim=ecg_embed_dim)
        self.face_encoder = FaceEncoder(embed_dim=face_embed_dim)

        total_embed = ecg_embed_dim + face_embed_dim
        # Build shared fusion MLP
        fusion_layers = []
        in_dim = total_embed
        for h_dim in fusion_hidden[1:]:
            fusion_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.fusion = nn.Sequential(*fusion_layers)

        # Task-specific heads (each: Linear → output logit)
        self.num_tasks = num_tasks
        self.heads = nn.ModuleList([
            nn.Linear(fusion_hidden[-1], 1) for _ in range(num_tasks)
        ])

    def forward(self, ecg, face, task_mask=None):
        """Forward pass.

        Args:
            ecg:  (B, 1, L) ECG signal tensor.
            face: (B, 1, H, W) Face image tensor.
            task_mask: (B, num_tasks) float mask, 1=valid label, 0=missing.
                       If None, all tasks assumed valid.

        Returns:
            logits: (B, num_tasks) raw logits per task.
        """
        ecg_emb = self.ecg_encoder(ecg)       # (B, ecg_embed_dim)
        face_emb = self.face_encoder(face)     # (B, face_embed_dim)

        fused = torch.cat([ecg_emb, face_emb], dim=-1)  # (B, total_embed)
        shared = self.fusion(fused)            # (B, fusion_hidden[-1])

        logits = []
        for head in self.heads:
            logits.append(head(shared))         # each (B, 1)
        logits = torch.cat(logits, dim=-1)      # (B, num_tasks)

        return logits

    def predict_proba(self, ecg, face):
        """Return predicted probabilities for all tasks.

        Returns:
            probs: (B, num_tasks) in [0, 1].
        """
        logits = self.forward(ecg, face)
        return torch.sigmoid(logits)


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
