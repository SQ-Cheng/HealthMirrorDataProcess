"""Keep the pretrained RGB filters and add a zero-initialized IR input slice."""

import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model as rgb_model


def build_model(architecture, weights_dir):
    if architecture != "efficientnet_b0":
        raise ValueError("This experiment uses only EfficientNet-B0")
    model, head, weight_path = rgb_model(architecture, weights_dir)
    old = model.features[0][0]
    if old.in_channels != 3 or old.groups != 1:
        raise ValueError("Unexpected pretrained stem")
    # Preserve the primary initialization/RNG sequence after allocating the new conv.
    with torch.random.fork_rng(devices=[]):
        stem = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                         old.dilation, old.groups, bias=old.bias is not None, padding_mode=old.padding_mode)
    with torch.no_grad():
        stem.weight[:, :3].copy_(old.weight)
        stem.weight[:, 3:].zero_()
        if old.bias is not None:
            stem.bias.copy_(old.bias)
    model.features[0][0] = stem
    return model, head, weight_path
