"""Model registry for Exp1-SQAPreTrain.

All architectures share the same interface:
    model(x_masked, visible_mask) -> reconstruction

Where:
    x_masked:     (B, C, L) — masked input signal (C=1)
    visible_mask: (B, C, L) — 1=visible, 0=masked
    reconstruction: (B, C, L) — predicted signal
"""

from .cnn import build_cnn_model
from .resnet import (
    build_resnet_decoder,
    build_resnet_encoder,
    build_resnet_model,
)
from .tcn import build_tcn_decoder, build_tcn_encoder, build_tcn_model

_MODEL_BUILDERS = {
    "cnn": build_cnn_model,
    "resnet": build_resnet_model,
    "tcn": build_tcn_model,
}


def build_model(model_name, target_length=256):
    """Build a model by name.

    Args:
        model_name: 'cnn', 'resnet', or 'tcn'.
        target_length: Input sequence length (e.g. 256, 512, 1024).

    Returns:
        nn.Module instance.
    """
    model_name = model_name.lower()
    if model_name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_MODEL_BUILDERS.keys())}"
        )
    return _MODEL_BUILDERS[model_name](target_length)


def list_models():
    """Return available model names."""
    return list(_MODEL_BUILDERS.keys())


__all__ = [
    "build_model",
    "list_models",
    "build_cnn_model",
    "build_resnet_model",
    "build_resnet_encoder",
    "build_resnet_decoder",
    "build_tcn_model",
    "build_tcn_encoder",
    "build_tcn_decoder",
]
