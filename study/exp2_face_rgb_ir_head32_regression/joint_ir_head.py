"""Joint IR-slice/head optimization with immutable pretrained RGB features."""

from contextlib import contextmanager
import json

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as functional

from study.exp2_face_pretrained_head32_regression import train as primary
from .models import build_model


IR_LR = 1e-4
HEAD_LR = 5e-5
MIN_LR = 1e-6
MAX_EPOCHS = 60
PATIENCE = 12


class FrozenRGBIRStem(nn.Module):
    """Concatenate frozen RGB and trainable IR filters for one four-channel conv."""

    def __init__(self, stem):
        super().__init__()
        if stem.in_channels != 4 or stem.groups != 1 or stem.padding_mode != "zeros":
            raise ValueError("Expected the standard four-channel EfficientNet stem")
        self.weight = nn.Parameter(stem.weight[:, :3].detach().clone(), requires_grad=False)
        self.ir_weight = nn.Parameter(stem.weight[:, 3:].detach().clone())
        self.bias = None if stem.bias is None else nn.Parameter(stem.bias.detach().clone(), requires_grad=False)
        self.stride, self.padding, self.dilation = stem.stride, stem.padding, stem.dilation
        self.groups, self.in_channels, self.out_channels = stem.groups, 4, stem.out_channels

    def forward(self, images):
        weights = torch.cat((self.weight, self.ir_weight), dim=1)
        return functional.conv2d(images, weights, self.bias, self.stride, self.padding, self.dilation, self.groups)


def freeze_rgb(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.features[0][0].ir_weight.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)


def build_joint_model(architecture, weights_dir, rgb_checkpoint, target_scaler):
    model, head, source = build_model(architecture, weights_dir, rgb_checkpoint, target_scaler)
    model.features[0][0] = FrozenRGBIRStem(model.features[0][0])
    freeze_rgb(model)
    return model, head, source


def joint_optimizer(parameters, lr=IR_LR, weight_decay=1e-4):
    parameters = list(parameters)
    ir = [parameter for parameter in parameters if tuple(parameter.shape) == (32, 1, 3, 3)]
    head = [parameter for parameter in parameters if all(parameter is not item for item in ir)]
    if len(ir) != 1 or sum(p.numel() for p in head) != 41089:
        raise ValueError("Only the 288 IR filters and head32 may enter the optimizer")
    if lr != IR_LR:
        raise ValueError("Unexpected joint-stage learning rate")
    return torch.optim.AdamW([
        {"params": ir, "lr": IR_LR, "name": "ir"},
        {"params": head, "lr": HEAD_LR, "name": "head"},
    ], weight_decay=weight_decay)


@contextmanager
def joint_training():
    names = ("unfreeze_all", "AdamW", "_train_epoch", "DIRECT_LEARNING_RATE", "DIRECT_PATIENCE")
    original = {name: getattr(primary, name) for name in names}

    def train_epoch(model, head, loader, optimizer, scaler, criterion, device, encoder_frozen, **kwargs):
        # Eval mode freezes backbone buffers/stochastic depth, but NOT IR autograd.
        return original["_train_epoch"](model, head, loader, optimizer, scaler, criterion, device, True, **kwargs)

    primary.unfreeze_all = freeze_rgb
    primary.AdamW = joint_optimizer
    primary._train_epoch = train_epoch
    primary.DIRECT_LEARNING_RATE = IR_LR
    primary.DIRECT_PATIENCE = PATIENCE
    print(f"[joint-ir-head] RGB params/BN/stochastic-depth=frozen; IR lr={IR_LR:.1e} "
          f"head lr={HEAD_LR:.1e}; trainable=41377; cosine floor={MIN_LR:.1e}", flush=True)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(primary, name, value)


def audit_and_annotate(checkpoint, path, source, weights_dir, target_scaler, max_epochs=MAX_EPOCHS):
    initial, _, _ = build_joint_model("efficientnet_b0", weights_dir, source, target_scaler)
    before, after = initial.state_dict(), checkpoint["model_state_dict"]
    changed_frozen = [key for key in before if key.startswith("features.")
                      and key != "features.0.0.ir_weight" and not torch.equal(before[key], after[key])]
    if changed_frozen:
        raise RuntimeError(f"Frozen RGB parameters/buffers changed: {changed_frozen}")
    norm = after["features.0.0.ir_weight"].norm().item()
    head_changed = any(not torch.equal(before[key], after[key]) for key in before if key.startswith("classifier."))
    history = pd.read_csv(path / "history.csv")
    if norm <= 0 or not head_changed or history.optimizer_steps.sum() <= 0:
        raise RuntimeError("Joint training did not update both IR filters and head")
    history["ir_learning_rate"] = history.learning_rate
    history["head_learning_rate"] = MIN_LR+(history.learning_rate-MIN_LR)*(HEAD_LR-MIN_LR)/(IR_LR-MIN_LR)
    history.to_csv(path / "history.csv", index=False)
    settings = {"training_protocol": "joint_ir_head_frozen_rgb", "single_stage": True,
        "trainable_parameters": 41377, "ir_trainable_parameters": 288, "head_trainable_parameters": 41089,
        "ir_learning_rate": IR_LR, "head_learning_rate": HEAD_LR, "minimum_learning_rate": MIN_LR,
        "max_epochs": max_epochs, "patience": PATIENCE, "batchnorm_running_stats_frozen": True,
        "backbone_stochastic_depth_enabled": False, "head_dropout": .25,
        "stem_parameterization": "features.0.0.weight=RGB3 frozen; ir_weight=IR1 trainable; concatenate in forward",
        "frozen_parameters_and_buffers_exactly_unchanged": True}
    checkpoint.update(settings)
    # Remove unused two-stage fields from the shared trainer's checkpoint metadata.
    for key in ("finetune_learning_rate", "head_min_learning_rate", "finetune_min_learning_rate", "head_max_epochs",
                "finetune_max_epochs", "head_patience", "finetune_patience", "head_warmup_epochs", "finetune_warmup_epochs"):
        checkpoint.pop(key, None)
    stage_path = path / "stage_direct_best.pt"
    stage = torch.load(stage_path, map_location="cpu", weights_only=True)
    stage.update(settings, input_channels=4)
    torch.save(stage, stage_path)
    audit = {**settings, "initial_ir_norm": 0.0, "selected_ir_norm": norm, "head_updated": head_changed,
             "verified_frozen_state_entries": sum(key.startswith("features.") and key != "features.0.0.ir_weight" for key in before)}
    (path / "frozen_rgb_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(f"[frozen-rgb-audit] {target_scaler.target}: all RGB parameters/buffers unchanged; IR norm={norm:.6g}; head updated", flush=True)
    return {"joint": norm, "selected": norm}
