"""Load both saved components for composed raw-unit inference."""

import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression.config import WEIGHTS_DIR
from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model
from study.exp2_history_only_head32_regression.models import HistoryOnlyRegressor


class HistoryFaceResidualRegressor(nn.Module):
    def __init__(self, checkpoint, weights_dir=WEIGHTS_DIR):
        super().__init__()
        self.history = HistoryOnlyRegressor()
        self.face, _, _ = build_pretrained_model("efficientnet_b0", weights_dir)
        self.history.load_state_dict(checkpoint["history_state_dict"], strict=True)
        self.face.load_state_dict(checkpoint["face_state_dict"], strict=True)
        self.median = checkpoint["lab_target_scaler"]["median"]
        self.iqr = checkpoint["lab_target_scaler"]["iqr"]
        self.history.requires_grad_(False)

    def forward(self, images, history, mask):
        return self.median + self.iqr * (self.history(history, mask) + self.face(images))

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls(checkpoint).to(device).eval()
