"""SQA model built from a pretrained ECG encoder and a lightweight head."""

import inspect

import torch
import torch.nn as nn

from ..models import (
    build_model,
    build_resnet_encoder,
    build_tcn_encoder,
)


def _first_conv1d(module):
    for child in module.modules():
        if isinstance(child, nn.Conv1d):
            return child
    raise ValueError("Encoder does not contain a Conv1d input layer.")


def _forward_argument_count(module):
    signature = inspect.signature(module.forward)
    return len([
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ])


def load_pretrained_encoder(architecture, checkpoint_path, target_length, map_location="cpu"):
    """Strictly load a pretrained reconstruction model and return only its encoder."""
    architecture = architecture.lower()
    if architecture not in {"resnet", "tcn"}:
        raise ValueError("architecture must be 'resnet' or 'tcn'")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    checkpoint_architecture = checkpoint.get("model")
    checkpoint_length = checkpoint.get("target_length")
    if checkpoint_architecture and checkpoint_architecture != architecture:
        raise ValueError(
            f"Checkpoint architecture is '{checkpoint_architecture}', expected '{architecture}'."
        )
    if checkpoint_length and int(checkpoint_length) != int(target_length):
        raise ValueError(
            f"Checkpoint target_length is {checkpoint_length}, expected {target_length}."
        )

    if "model_state_dict" in checkpoint:
        reconstruction_model = build_model(architecture, target_length=target_length)
        reconstruction_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return reconstruction_model.encoder, checkpoint

    if "encoder_state_dict" in checkpoint:
        builder = build_resnet_encoder if architecture == "resnet" else build_tcn_encoder
        encoder = builder(target_length=target_length)
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        return encoder, checkpoint

    raise KeyError("Checkpoint must contain 'model_state_dict' or 'encoder_state_dict'.")


class ECGSQAModel(nn.Module):
    """Frozen ECG encoder plus global pooling and a two-task sigmoid-logit head."""

    def __init__(self, encoder, hidden_dim=128, dropout=0.2, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)

        input_conv = _first_conv1d(encoder)
        self.encoder_input_channels = int(input_conv.in_channels)
        self.encoder_forward_args = _forward_argument_count(encoder)
        if self.encoder_input_channels not in {1, 2}:
            raise ValueError(
                "SQA supports encoders expecting one ECG channel or two "
                f"[ECG, mask] channels, got {self.encoder_input_channels}."
            )

        output_channels = getattr(encoder, "out_channels", None)
        if output_channels is None:
            raise ValueError("Encoder must expose its latent 'out_channels'.")
        self.output_channels = int(output_channels)

        self.head = nn.Sequential(
            nn.Linear(2 * self.output_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.set_encoder_frozen(self.freeze_encoder)

    def set_encoder_frozen(self, frozen=True):
        self.freeze_encoder = bool(frozen)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not self.freeze_encoder
        if self.freeze_encoder:
            self.encoder.eval()

    def _encode(self, ecg):
        if ecg.ndim != 3 or ecg.shape[1] != 1:
            raise ValueError(f"Expected ECG shape [B, 1, T], got {tuple(ecg.shape)}")

        if self.encoder_input_channels == 1:
            return self.encoder(ecg)

        zero_mask = torch.zeros_like(ecg)
        if self.encoder_forward_args >= 2:
            # Current pretrained encoders concatenate [ECG, mask] internally.
            return self.encoder(ecg, zero_mask)
        return self.encoder(torch.cat([ecg, zero_mask], dim=1))

    def forward(self, ecg):
        latent = self._encode(ecg)
        if latent.ndim != 3:
            raise ValueError(f"Expected encoder output [B, C, T], got {tuple(latent.shape)}")
        average = latent.mean(dim=-1)
        maximum = latent.amax(dim=-1)
        return self.head(torch.cat([average, maximum], dim=1))

    def predict_proba(self, ecg):
        return torch.sigmoid(self(ecg))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self
