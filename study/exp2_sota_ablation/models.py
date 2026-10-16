"""Scalar modality fusion, or hierarchical scalar expert/modality fusion."""

import math

import torch
from torch import nn

from study.exp2_face_dinov3_farl_regression.models import ScalarGatedHead
from study.exp2_face_rgb_ir_dinov3b_regression.models import build_model as baseline_model


class Fusion(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.register_buffer("feature_mean", torch.zeros(2560))
        self.register_buffer("feature_std", torch.ones(2560))
        if settings["model"] == "scalar":
            self.network = baseline_model("dino_farl_scalar_gate")
        else:
            self.rgb_expert = ScalarGatedHead(dino_features=768)
            self.ir_expert = ScalarGatedHead(dino_features=768)
            # Keep one task head, not the two unused expert heads.
            self.head = self.rgb_expert.head
            self.rgb_expert.head = nn.Identity()
            self.ir_expert.head = nn.Identity()
            self.modality_gate = nn.Linear(128, 1)
            nn.init.zeros_(self.modality_gate.weight)
        gate = self.network.gate if settings["model"] == "scalar" else self.modality_gate
        nn.init.constant_(gate.bias, math.log(settings["rgb_prior"]/(1-settings["rgb_prior"])))
        head = self.network.head if settings["model"] == "scalar" else self.head
        head[3].p = settings["dropout"]
        self.projection_dropout = nn.Dropout(settings["projection_dropout"])

    def forward(self, features):
        x = (features-self.feature_mean)/self.feature_std
        rgb, ir = x.split(1280, dim=1)
        if self.settings["model"] == "scalar":
            rgb, ir = self.network.rgb_project(rgb), self.network.ir_project(ir)
            head, gate = self.network.head, self.network.gate
        else:
            rgb, ir = self.rgb_expert(rgb), self.ir_expert(ir)
            head, gate = self.head, self.modality_gate
        rgb, ir = self.projection_dropout(rgb), self.projection_dropout(ir)
        if self.training and self.settings["ir_dropout"]:
            if len(ir) % 20:
                raise ValueError("Modality dropout requires complete video/view groups")
            keep = (torch.rand(len(ir)//20, 1, device=ir.device) >= self.settings["ir_dropout"]).repeat_interleave(20, 0)
            ir = ir*keep
        weight = torch.sigmoid(gate(torch.cat((rgb, ir), dim=1)))
        return head(weight*rgb+(1-weight)*ir).flatten()

    def warm_start_rgb(self, checkpoint, target_scaler):
        if self.settings["model"] != "hierarchical" or self.settings["normalize"]:
            raise ValueError("Warm start requires the unnormalized hierarchical model")
        state = checkpoint["state_dict"]
        expert = {key: value for key, value in state.items() if not key.startswith("head.")}
        self.rgb_expert.load_state_dict(expert, strict=True)
        self.ir_expert.load_state_dict(expert, strict=True)
        self.head.load_state_dict({key.removeprefix("head."): value for key, value in state.items() if key.startswith("head.")}, strict=True)
        source = checkpoint["target_scaler"]
        if source["unit"] != target_scaler["unit"]:
            raise ValueError("RGB warm-start target units differ")
        with torch.no_grad():
            self.head[-1].weight.mul_(source["iqr"]/target_scaler["iqr"])
            self.head[-1].bias.mul_(source["iqr"]/target_scaler["iqr"]).add_((source["median"]-target_scaler["median"])/target_scaler["iqr"])
