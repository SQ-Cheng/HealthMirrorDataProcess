"""The same frozen encoder is applied to both timepoints; heads are independent."""

import torch
from torch import nn

from study.exp2_face_dinov3_frozen.backbone import Head


class FrozenDeltaPredictor(nn.Module):
    def __init__(self, encoder, hidden):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.head = Head(hidden)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, first, second):
        with torch.no_grad():
            features = self.encoder(torch.cat((first, second)))
            early, late = features.chunk(2)
        return self.head(late - early)


def head_parameters(hidden):
    return 388 * hidden + 1
