"""Keep the pretrained RGB filters and add a zero-initialized IR input slice."""

import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model as rgb_model
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler


def build_model(architecture, weights_dir, rgb_checkpoint=None, target_scaler=None):
    if architecture != "efficientnet_b0":
        raise ValueError("This experiment uses only EfficientNet-B0")
    model, head, weight_path = rgb_model(architecture, weights_dir)
    if rgb_checkpoint is not None:
        checkpoint = torch.load(rgb_checkpoint, map_location="cpu", weights_only=True)
        if (target_scaler is None or checkpoint["architecture"] != architecture
                or checkpoint["target"] != target_scaler.target
                or checkpoint["task_type"] != "robust_scaled_raw_value_regression"):
            raise ValueError("RGB checkpoint target/architecture/scaling mismatch")
        source_scaler = RobustTargetScaler(**checkpoint["target_scaler"])
        if source_scaler.unit != target_scaler.unit or min(source_scaler.iqr, target_scaler.iqr) <= 0:
            raise ValueError("Incompatible RGB checkpoint units or scales")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Preserve raw predictions when the paired training cohort has a different scaler.
        factor = source_scaler.iqr / target_scaler.iqr
        offset = (source_scaler.median - target_scaler.median) / target_scaler.iqr
        with torch.no_grad():
            head[-1].weight.mul_(factor)
            head[-1].bias.mul_(factor).add_(offset)
        weight_path = str(rgb_checkpoint)
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
