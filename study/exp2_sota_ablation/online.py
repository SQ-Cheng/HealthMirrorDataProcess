"""Adapt only the final IR Transformer blocks; RGB uses immutable cached features."""

import torch
from torch import nn

from study.exp2_face_dinov3_frozen.backbone import load_encoder
from study.exp2_face_dinov3_farl_regression.backbone import load_farl
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from study.exp2_face_pretrained_head32_regression import config as image_config
from . import config
from study.common.ir_compile import compile_ir, prepare_farl, POLICY


def select_tail(dino, farl):
    dino.requires_grad_(False)
    farl.requires_grad_(False)
    for module in (dino.blocks[-1], dino.norm, farl.transformer.resblocks[-1], farl.ln_post):
        module.requires_grad_(True)
    farl.proj.requires_grad_(True)


class OnlineIR(nn.Module):
    def __init__(self, fusion, settings, device):
        super().__init__()
        self.fusion, self.settings = fusion, settings
        self.dino = load_encoder(architecture="dinov3_vitb16").to(device)
        self.farl = load_farl()[0].to(device)
        prepare_farl(self.farl)
        select_tail(self.dino, self.farl)
        self.dino.eval()
        self.farl.eval()
        for name, encoder in (("dino_forward", self.dino), ("farl_forward", self.farl)):
            if settings["compile_online"]:
                encoder = compile_ir(encoder)
            # Do not register duplicate encoder modules/state-dict entries.
            object.__setattr__(self, name, encoder)
        print(f"[ir-compile] enabled={settings['compile_online']} policy={POLICY} microbatch={settings['online_microbatch']}", flush=True)

    def train(self, mode=True):
        super().train(mode)
        # LayerNorm still trains; disable backbone stochastic regularizers.
        self.dino.eval()
        self.farl.eval()
        return self

    def forward_video(self, features, pixels, views, device):
        if len(features) != 20 or not views.eq(views[0]).all():
            raise ValueError("Online loss requires one complete twenty-frame view")
        if not torch.equal(pixels[:, 0], pixels[:, 1]) or not torch.equal(pixels[:, 1], pixels[:, 2]):
            raise ValueError("IR frames must be genuine replicated grayscale")
        adapted_features = []
        for start in range(0, 20, self.settings["online_microbatch"]):
            end = min(start+self.settings["online_microbatch"], 20)
            dino = _prepare_images(pixels[start:end], views[start:end], "bicubic", device,
                normalization_mean=image_config.IMAGENET_MEAN, normalization_std=image_config.IMAGENET_STD)
            farl = _prepare_images(pixels[start:end], views[start:end], "bicubic", device,
                normalization_mean=config.reference.farl_config.CLIP_MEAN, normalization_std=config.reference.farl_config.CLIP_STD)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                ir_dino, ir_farl = self.dino_forward(dino), self.farl_forward(farl)
                adapted_features.append(torch.cat((features[start:end, :1280].to(device), ir_dino, ir_farl), dim=1))
        # Preserve one modality-dropout decision per complete twenty-frame video.
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return self.fusion(torch.cat(adapted_features, dim=0))

    def tail_state(self):
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()
                if name.startswith(("dino.", "farl.")) and value.requires_grad}

    def load_tail(self, state):
        expected = {name: value for name, value in self.named_parameters()
                    if name.startswith(("dino.", "farl.")) and value.requires_grad}
        if set(state) != set(expected):
            raise ValueError("IR tail checkpoint parameter names differ")
        with torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value.to(expected[name]))
