"""Native-video and uniform-ECG regressor with official Mamba selective SSM."""

import importlib.metadata

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .config import (
    D_CONV,
    D_MODEL,
    D_STATE,
    DROPOUT,
    ECG_FIRST_STRIDE,
    ECG_SECOND_STRIDE,
    ECG_TOTAL_STRIDE,
    EXPAND,
    HEAD_HIDDEN_FEATURES,
    MAMBA_LAYERS,
    MAMBA_SSM_VERSION,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    WINDOW_SECONDS,
)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.GroupNorm(4 if input_channels < 32 else 8, input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class NativeFrameEncoder(nn.Module):
    """Encode every native 128x128 RGB frame without an input resize."""

    def __init__(self, output_features=D_MODEL):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            DepthwiseSeparableBlock(16, 32, stride=2),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, output_features, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, frames):
        if tuple(frames.shape[-2:]) != (VIDEO_HEIGHT, VIDEO_WIDTH):
            raise ValueError(
                f"Expected native {VIDEO_HEIGHT}x{VIDEO_WIDTH} input, "
                f"got {tuple(frames.shape[-2:])}"
            )
        return self.features(frames).flatten(1)


class RawEcgTokenizer(nn.Module):
    """Learn local morphology tokens from a uniformly sampled ECG waveform."""

    def __init__(self, output_features=D_MODEL):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                kernel_size=9,
                stride=ECG_FIRST_STRIDE,
                padding=4,
                bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(
                32,
                32,
                kernel_size=9,
                stride=ECG_SECOND_STRIDE,
                padding=4,
                groups=32,
                bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, output_features, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_features),
            nn.SiLU(),
        )

    def forward(self, ecg):
        return self.layers(ecg.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def output_lengths(input_lengths):
        first = torch.div(
            input_lengths + ECG_FIRST_STRIDE - 1,
            ECG_FIRST_STRIDE,
            rounding_mode="floor",
        )
        return torch.div(
            first + ECG_SECOND_STRIDE - 1,
            ECG_SECOND_STRIDE,
            rounding_mode="floor",
        )


class ResidualMambaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(D_MODEL)
        self.mamba = Mamba(
            d_model=D_MODEL,
            d_state=D_STATE,
            d_conv=D_CONV,
            expand=EXPAND,
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, inputs):
        return inputs + self.dropout(self.mamba(self.norm(inputs)))


class VideoEcgMambaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_encoder = NativeFrameEncoder()
        self.ecg_tokenizer = RawEcgTokenizer()
        self.modality_embedding = nn.Embedding(3, D_MODEL)
        self.time_projection = nn.Sequential(
            nn.Linear(3, D_MODEL),
            nn.SiLU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.summary_token = nn.Parameter(torch.zeros(1, D_MODEL))
        nn.init.normal_(self.summary_token, std=0.02)
        self.backbone = nn.ModuleList(
            [ResidualMambaBlock() for _ in range(MAMBA_LAYERS)]
        )
        self.final_norm = nn.RMSNorm(D_MODEL)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, HEAD_HIDDEN_FEATURES),
            nn.LayerNorm(HEAD_HIDDEN_FEATURES),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_HIDDEN_FEATURES, 1),
        )

    def _encode_video(self, frames, lengths):
        batch, steps = frames.shape[:2]
        valid = (
            torch.arange(steps, device=frames.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        normalized = frames[valid].float().div_(127.5).sub_(1.0)
        encoded = self.frame_encoder(normalized)
        output = encoded.new_zeros((batch, steps, D_MODEL))
        output[valid] = encoded
        return output

    def _ecg_token_times(self, ecg_times, input_lengths, output_lengths):
        token_times = []
        for batch_index in range(len(input_lengths)):
            output_length = int(output_lengths[batch_index])
            indices = torch.arange(
                output_length, device=ecg_times.device, dtype=torch.long
            ) * ECG_TOTAL_STRIDE
            indices.clamp_(max=int(input_lengths[batch_index]) - 1)
            token_times.append(ecg_times[batch_index, indices])
        return token_times

    def _time_features(self, times):
        normalized = (times / WINDOW_SECONDS).clamp(0.0, 1.0)
        phase = 2.0 * torch.pi * normalized
        return torch.stack((normalized, torch.sin(phase), torch.cos(phase)), dim=-1)

    def _merge_modalities(
        self,
        video_tokens,
        frame_times,
        frame_lengths,
        ecg_tokens,
        ecg_times,
        ecg_lengths,
    ):
        ecg_output_lengths = self.ecg_tokenizer.output_lengths(ecg_lengths)
        ecg_token_times = self._ecg_token_times(
            ecg_times, ecg_lengths, ecg_output_lengths
        )
        sequences, summary_indices = [], []
        for batch_index in range(len(frame_lengths)):
            video_length = int(frame_lengths[batch_index])
            ecg_length = int(ecg_output_lengths[batch_index])
            tokens = torch.cat(
                (
                    video_tokens[batch_index, :video_length],
                    ecg_tokens[batch_index, :ecg_length],
                ),
                dim=0,
            )
            times = torch.cat(
                (
                    frame_times[batch_index, :video_length],
                    ecg_token_times[batch_index],
                )
            )
            modalities = torch.cat(
                (
                    torch.zeros(
                        video_length, dtype=torch.long, device=tokens.device
                    ),
                    torch.ones(ecg_length, dtype=torch.long, device=tokens.device),
                )
            )
            order = torch.argsort(times, stable=True)
            tokens = tokens[order]
            times = times[order]
            modalities = modalities[order]
            tokens = (
                tokens
                + self.modality_embedding(modalities)
                + self.time_projection(self._time_features(times))
            )
            summary = (
                self.summary_token
                + self.modality_embedding.weight[2].unsqueeze(0)
                + self.time_projection(
                    self._time_features(times.new_tensor([WINDOW_SECONDS]))
                )
            )
            summary_indices.append(len(tokens))
            sequences.append(torch.cat((tokens, summary), dim=0))
        padded = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )
        return padded.contiguous(), torch.tensor(
            summary_indices, device=padded.device, dtype=torch.long
        )

    def forward(
        self,
        frames,
        frame_times,
        frame_lengths,
        ecg,
        ecg_times,
        ecg_lengths,
    ):
        video_tokens = self._encode_video(frames, frame_lengths)
        ecg_tokens = self.ecg_tokenizer(ecg)
        tokens, summary_indices = self._merge_modalities(
            video_tokens,
            frame_times,
            frame_lengths,
            ecg_tokens,
            ecg_times,
            ecg_lengths,
        )
        for block in self.backbone:
            tokens = block(tokens)
        summary = tokens[
            torch.arange(len(tokens), device=tokens.device), summary_indices
        ]
        return self.head(self.final_norm(summary)).squeeze(-1)


def dependency_versions():
    return {
        "mamba_ssm": importlib.metadata.version("mamba-ssm"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def build_model():
    version = importlib.metadata.version("mamba-ssm")
    if version != MAMBA_SSM_VERSION:
        raise RuntimeError(
            f"Expected mamba-ssm {MAMBA_SSM_VERSION}, found {version}. "
            "Run install_mamba.sh."
        )
    return VideoEcgMambaRegressor()


def parameter_counts(model):
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "mamba": sum(
            parameter.numel()
            for block in model.backbone
            for parameter in block.parameters()
        ),
        "frame_encoder": sum(
            parameter.numel() for parameter in model.frame_encoder.parameters()
        ),
        "ecg_tokenizer": sum(
            parameter.numel() for parameter in model.ecg_tokenizer.parameters()
        ),
        "head": sum(parameter.numel() for parameter in model.head.parameters()),
    }


__all__ = [
    "VideoEcgMambaRegressor",
    "build_model",
    "dependency_versions",
    "parameter_counts",
]
