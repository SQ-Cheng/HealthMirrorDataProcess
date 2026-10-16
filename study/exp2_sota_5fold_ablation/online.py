"""Microbatch IR encoders, but run stochastic fusion on the complete video."""

import torch

from study.exp2_sota_ablation.online import OnlineIR as BaseOnlineIR
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from study.exp2_face_pretrained_head32_regression import config as image_config
from . import config


class OnlineIR(BaseOnlineIR):
    def forward_video(self, features, pixels, views, device):
        if len(features) != 20 or not views.eq(views[0]).all():
            raise ValueError("Online loss requires one complete twenty-frame view")
        if not torch.equal(pixels[:, 0], pixels[:, 1]) or not torch.equal(pixels[:, 1], pixels[:, 2]):
            raise ValueError("IR must be replicated grayscale")
        adapted = []
        for start in range(0, 20, self.settings["online_microbatch"]):
            end = min(start+self.settings["online_microbatch"], 20)
            dino = _prepare_images(pixels[start:end], views[start:end], "bicubic", device,
                normalization_mean=image_config.IMAGENET_MEAN, normalization_std=image_config.IMAGENET_STD)
            farl = _prepare_images(pixels[start:end], views[start:end], "bicubic", device,
                normalization_mean=config.parent.reference.farl_config.CLIP_MEAN,
                normalization_std=config.parent.reference.farl_config.CLIP_STD)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                adapted.append(torch.cat((features[start:end, :1280].to(device),
                    self.dino_forward(dino), self.farl_forward(farl)), dim=1))
        # Modality dropout samples once per20-frame video, not per4-frame encoder call.
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return self.fusion(torch.cat(adapted, dim=0))
