import math
import torch
import torch.nn as nn


class ConvFeatureExtractor(nn.Module):
    """1D CNN feature extractor for one signal modality."""

    def __init__(self, in_channels=1, hidden_channels=(32, 64, 96), dropout=0.1):
        super().__init__()
        c1, c2, c3 = hidden_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Input: (B, 1, L) -> Output: (B, C, L/8)
        return self.net(x)


class CNNTransformerBP(nn.Module):
    """
    CNN + Transformer model for BP prediction from ECG + rPPG.

    Input:
        ecg:  (B, 1, L)
        rppg: (B, 1, L)
    Output:
        (B, 2) where channels are [SBP, DBP] in normalized [0,1] target space.
    """

    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=3,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.ecg_encoder = ConvFeatureExtractor(dropout=dropout)
        self.rppg_encoder = ConvFeatureExtractor(dropout=dropout)

        fused_channels = 96 * 2
        self.input_proj = nn.Linear(fused_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _sinusoidal_positional_encoding(length, dim, device):
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, ecg, rppg):
        ecg_feat = self.ecg_encoder(ecg)    # (B, 96, T)
        rppg_feat = self.rppg_encoder(rppg)  # (B, 96, T)

        fused = torch.cat([ecg_feat, rppg_feat], dim=1)  # (B, 192, T)
        fused = fused.permute(0, 2, 1)  # (B, T, 192)
        tokens = self.input_proj(fused)  # (B, T, d_model)

        bsz = tokens.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)

        pos = self._sinusoidal_positional_encoding(x.size(1), x.size(2), x.device)
        x = x + pos.unsqueeze(0)

        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)
