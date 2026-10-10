"""Load every FaRL image trunk parameter strictly, omit only known MIM branches."""

import torch
from clip.model import VisionTransformer
from . import config


def load_farl():
    checkpoint = torch.load(config.WEIGHTS, map_location="cpu", weights_only=True)
    state = {key.removeprefix("visual."): value for key, value in checkpoint["state_dict"].items() if key.startswith("visual.")}
    encoder = VisionTransformer(224, 16, 768, 12, 12, 512)
    expected = set(encoder.state_dict())
    extra = set(state) - expected
    allowed = lambda key: key == "mask_token" or key.startswith(("lm_transformer.", "lm_head.", "ln_lm."))
    if expected - set(state) or any(not allowed(key) for key in extra):
        raise RuntimeError("FaRL image checkpoint has unexpected missing/extra inference parameters")
    encoder.load_state_dict({key: state[key] for key in expected}, strict=True)
    return encoder.eval().requires_grad_(False), sorted(extra)
