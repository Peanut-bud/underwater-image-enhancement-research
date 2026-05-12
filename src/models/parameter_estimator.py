"""Parameter estimator for airlight and transmission."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ParameterEstimator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.transmission_head = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        airlight = self.airlight_head(features)
        transmission = self.transmission_head(features)
        return airlight, transmission
