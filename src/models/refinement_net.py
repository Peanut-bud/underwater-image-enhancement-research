"""Lightweight refinement subnet inspired by NAF-style residual blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class NAFLikeBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.expand(x)
        x1, x2 = torch.chunk(self.depthwise(x), chunks=2, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = self.project(x)
        return residual + x


class RefinementNet(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 32, blocks: int = 2, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[NAFLikeBlock(hidden_channels) for _ in range(blocks)])
        self.head = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, rough: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(features)
        hidden = self.blocks(hidden)
        residual = torch.tanh(self.head(hidden))
        return (rough + self.residual_scale * residual).clamp(0.0, 1.0)
