"""Model registry for Exp1-SQAPreTrain.

All models follow the same interface:
    model(x_masked, visible_mask) -> reconstruction

Where:
    x_masked:   (B, C, L) — masked input signal
    visible_mask: (B, C, L) — 1=visible, 0=masked
    reconstruction: (B, C, L) — predicted signal

For single-signal models: C=1 (ecg or rppg).
For joint models: C=2 (ecg + rppg).
"""

from .baseline import build_baseline_model
from .tcn import build_tcn_model
from .mamba import build_mamba_model
from .transformer import build_transformer_model
from .gan import build_gan_generator, build_gan_discriminator
from .joint import build_joint_model


_MODEL_BUILDERS = {
    "baseline": build_baseline_model,
    "tcn": build_tcn_model,
    "mamba": build_mamba_model,
    "transformer": build_transformer_model,
    "gan": build_gan_generator,
    "joint": build_joint_model,
}

_DEFAULT_VARIANTS = {
    "baseline": "light",
    "tcn": "tcn256",
    "mamba": "light",
    "transformer": "light",
    "gan": "light",
    "joint": "light",
}


def build_model(model_name, variant=None):
    """Build a model by name.

    Args:
        model_name: One of 'baseline', 'tcn', 'mamba', 'transformer', 'gan', 'joint'.
        variant: Model-specific variant (e.g., 'light', 'full', 'tcn256', 'tcn512').

    Returns:
        nn.Module instance.
    """
    model_name = model_name.lower()
    if model_name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_BUILDERS.keys())}"
        )
    if variant is None:
        variant = _DEFAULT_VARIANTS[model_name]
    return _MODEL_BUILDERS[model_name](variant)


def list_models():
    """Return available model names."""
    return list(_MODEL_BUILDERS.keys())


__all__ = [
    "build_model",
    "list_models",
    "build_baseline_model",
    "build_tcn_model",
    "build_mamba_model",
    "build_transformer_model",
    "build_gan_generator",
    "build_gan_discriminator",
    "build_joint_model",
]
