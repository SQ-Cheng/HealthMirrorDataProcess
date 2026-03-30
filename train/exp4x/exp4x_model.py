"""Experiment 04-X models: three SQI regressors (exp4-1/2/3)."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Exp4Model1(nn.Module):
    """Compact CNN regressor baseline."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, kernel_size=9, stride=2),
            ConvBlock(16, 24, kernel_size=7, stride=2),
            ConvBlock(24, 32, kernel_size=5, stride=2),
            ConvBlock(32, 48, kernel_size=5, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x).squeeze(1)


class Exp4Model2(nn.Module):
    """CNN encoder + BiGRU temporal regressor."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(1, 16, kernel_size=9, stride=2),
            ConvBlock(16, 24, kernel_size=7, stride=2),
            ConvBlock(24, 32, kernel_size=5, stride=2),
        )
        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.transpose(1, 2)
        seq, _ = self.rnn(x)
        pooled = seq.mean(dim=1)
        return self.head(pooled).squeeze(1)


class Exp4Model3(nn.Module):
    """Dilated multi-scale temporal CNN regressor."""

    def __init__(self):
        super().__init__()
        self.proj = ConvBlock(1, 24, kernel_size=7, stride=2)
        self.ms = nn.Sequential(
            ConvBlock(24, 32, kernel_size=3, dilation=1),
            ConvBlock(32, 32, kernel_size=3, dilation=2),
            ConvBlock(32, 48, kernel_size=3, dilation=4),
            ConvBlock(48, 48, kernel_size=3, dilation=8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(48, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.ms(x)
        return self.head(x).squeeze(1)


def build_exp4x_model(model_name="exp4-1"):
    name = model_name.lower()
    if name == "exp4-1":
        return Exp4Model1()
    if name == "exp4-2":
        return Exp4Model2()
    if name == "exp4-3":
        return Exp4Model3()
    raise ValueError("model_name must be one of: exp4-1, exp4-2, exp4-3")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
