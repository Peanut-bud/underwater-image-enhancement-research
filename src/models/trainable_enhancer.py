"""Trainable stage-one backbone for underwater enhancement.

This model keeps the same high-level order as the deterministic stage-one
enhancer while replacing fixed operations with learnable modules:

input -> shallow features -> highlight-aware modulation
      -> frequency enhancement -> differential detail -> reconstruction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrainableShallowFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=3),
            ConvBlock(base_channels, base_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HighlightAwareBranch(nn.Module):
    """Predict a lightweight highlight map from shallow features.

    The branch is intentionally simple in stage one: it learns where bright,
    over-exposed, or reflection-heavy regions are likely to be, then provides
    a soft mask used to suppress over-enhancement in those areas.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = max(channels // 2, 8)
        self.layers = nn.Sequential(
            ConvBlock(channels, reduced, kernel_size=3),
            nn.Conv2d(reduced, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TrainableFrequencyEnhancementBlock(nn.Module):
    """Learnable approximation of frequency-aware color/detail enhancement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.high_proj = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.fuse = ConvBlock(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high_freq = x - low_freq

        low_enhanced = low_freq * self.low_proj(low_freq)
        high_enhanced = self.high_proj(high_freq)
        fused = torch.cat([low_enhanced, high_enhanced], dim=1)
        return self.fuse(fused) + x


class LearnableDiffBranch(nn.Module):
    """A trainable branch initialized from a classical differential kernel."""

    def __init__(self, channels: int, kernel: torch.Tensor) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self._init_kernel(kernel)

    def _init_kernel(self, kernel: torch.Tensor) -> None:
        with torch.no_grad():
            weight = kernel.view(1, 1, 3, 3).repeat(self.depthwise.out_channels, 1, 1, 1)
            self.depthwise.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pointwise(self.depthwise(x)))


class TrainableDifferentialDetailBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )

        self.branch_center = LearnableDiffBranch(channels, laplacian)
        self.branch_x = LearnableDiffBranch(channels, sobel_x)
        self.branch_y = LearnableDiffBranch(channels, sobel_y)
        self.fuse = nn.Sequential(
            ConvBlock(channels * 3, channels, kernel_size=1),
            ConvBlock(channels, channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center = self.branch_center(x)
        grad_x = self.branch_x(x)
        grad_y = self.branch_y(x)
        fused = self.fuse(torch.cat([center, grad_x, grad_y], dim=1))
        return fused + x


class TrainableReconstructionHead(nn.Module):
    def __init__(self, channels: int, out_channels: int = 3, residual_scale: float = 0.20) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.layers = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, original: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        residual = torch.tanh(self.layers(features))
        return torch.clamp(original + self.residual_scale * residual, 0.0, 1.0)


class TrainableEnhancer(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, highlight_suppression: float = 0.35) -> None:
        super().__init__()
        self.highlight_suppression = highlight_suppression
        self.shallow = TrainableShallowFeatureExtractor(in_channels=in_channels, base_channels=base_channels)
        self.highlight_branch = HighlightAwareBranch(base_channels)
        self.frequency = TrainableFrequencyEnhancementBlock(base_channels)
        self.detail = TrainableDifferentialDetailBlock(base_channels)
        self.reconstruction = TrainableReconstructionHead(base_channels, out_channels=in_channels)

    def compute_highlight_map(self, features: torch.Tensor) -> torch.Tensor:
        return self.highlight_branch(features)

    def apply_highlight_modulation(self, features: torch.Tensor, highlight_map: torch.Tensor) -> torch.Tensor:
        modulation = 1.0 - self.highlight_suppression * highlight_map
        return features * modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shallow(x)
        highlight_map = self.compute_highlight_map(features)
        features = self.apply_highlight_modulation(features, highlight_map)
        features = self.frequency(features)
        features = self.detail(features)
        return self.reconstruction(x, features)


def build_trainable_enhancer(base_channels: int = 32, highlight_suppression: float = 0.35) -> TrainableEnhancer:
    return TrainableEnhancer(base_channels=base_channels, highlight_suppression=highlight_suppression)
