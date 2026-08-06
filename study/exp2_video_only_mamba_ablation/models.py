"""Video-only forward path with shared initialization matching the paired model."""

import importlib.metadata

import torch
import torch.nn as nn

from study.exp2_video_ecg_mamba.models import VideoEcgMambaRegressor

from .config import D_MODEL, MAMBA_SSM_VERSION, WINDOW_SECONDS


class VideoOnlyMambaRegressor(VideoEcgMambaRegressor):
    """Remove ECG tokens while preserving all shared module initialization."""

    def __init__(self):
        super().__init__()
        for parameter in self.ecg_tokenizer.parameters():
            parameter.requires_grad_(False)

    def _merge_video(self, video_tokens, frame_times, frame_lengths):
        sequences, summary_indices = [], []
        for batch_index in range(len(frame_lengths)):
            length = int(frame_lengths[batch_index])
            times = frame_times[batch_index, :length]
            modalities = torch.zeros(
                length, dtype=torch.long, device=video_tokens.device
            )
            tokens = (
                video_tokens[batch_index, :length]
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
            summary_indices.append(length)
            sequences.append(torch.cat((tokens, summary), dim=0))
        padded = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )
        return padded.contiguous(), torch.tensor(
            summary_indices, device=padded.device, dtype=torch.long
        )

    def forward(self, frames, frame_times, frame_lengths):
        video_tokens = self._encode_video(frames, frame_lengths)
        tokens, summary_indices = self._merge_video(
            video_tokens, frame_times, frame_lengths
        )
        for block in self.backbone:
            tokens = block(tokens)
        summary = tokens[
            torch.arange(len(tokens), device=tokens.device), summary_indices
        ]
        return self.head(self.final_norm(summary)).squeeze(-1)


def build_model():
    version = importlib.metadata.version("mamba-ssm")
    if version != MAMBA_SSM_VERSION:
        raise RuntimeError(
            f"Expected mamba-ssm {MAMBA_SSM_VERSION}, found {version}"
        )
    return VideoOnlyMambaRegressor()


def parameter_counts(model):
    return {
        "total_stored": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "inactive_frozen_ecg_tokenizer": sum(
            parameter.numel() for parameter in model.ecg_tokenizer.parameters()
        ),
        "mamba": sum(
            parameter.numel()
            for block in model.backbone
            for parameter in block.parameters()
        ),
        "frame_encoder": sum(
            parameter.numel() for parameter in model.frame_encoder.parameters()
        ),
        "head": sum(parameter.numel() for parameter in model.head.parameters()),
    }


__all__ = ["VideoOnlyMambaRegressor", "build_model", "parameter_counts"]
