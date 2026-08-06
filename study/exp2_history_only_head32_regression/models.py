"""History encoder and width-32 regression head without an image pathway."""

import torch.nn as nn

from .config import (
    HEAD_HIDDEN_FEATURES,
    HISTORY_HIDDEN_FEATURES,
    HISTORY_INPUT_FEATURES,
    HISTORY_OUTPUT_FEATURES,
)


class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.measurement_mlp = nn.Sequential(
            nn.Linear(HISTORY_INPUT_FEATURES, HISTORY_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Linear(HISTORY_HIDDEN_FEATURES, HISTORY_OUTPUT_FEATURES),
            nn.LayerNorm(HISTORY_OUTPUT_FEATURES),
            nn.SiLU(inplace=True),
        )

    def forward(self, history, history_mask):
        encoded = self.measurement_mlp(history)
        mask = history_mask.unsqueeze(-1).to(encoded.dtype)
        count = mask.sum(dim=1).clamp_min_(1.0)
        return (encoded * mask).sum(dim=1) / count


class HistoryOnlyRegressor(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.history_encoder = HistoryEncoder()
        self.regressor = nn.Sequential(
            nn.Linear(HISTORY_OUTPUT_FEATURES, HEAD_HIDDEN_FEATURES),
            nn.LayerNorm(HEAD_HIDDEN_FEATURES),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN_FEATURES, 1),
        )

    def forward(self, history, history_mask):
        return self.regressor(self.history_encoder(history, history_mask))


def parameter_counts(model):
    history = sum(parameter.numel() for parameter in model.history_encoder.parameters())
    head = sum(parameter.numel() for parameter in model.regressor.parameters())
    return {"history_encoder": history, "head": head, "total": history + head}
