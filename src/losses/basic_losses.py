"""Basic losses for the first trainable stage.

Stage one intentionally keeps the loss stack simple:
- L1 loss
- SSIM loss
- Edge loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1ReconstructionLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """A compact differentiable SSIM loss implementation."""

    def __init__(self, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2) -> None:
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        padding = self.window_size // 2
        mu_x = F.avg_pool2d(pred, kernel_size=self.window_size, stride=1, padding=padding)
        mu_y = F.avg_pool2d(target, kernel_size=self.window_size, stride=1, padding=padding)

        sigma_x = F.avg_pool2d(pred * pred, kernel_size=self.window_size, stride=1, padding=padding) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, kernel_size=self.window_size, stride=1, padding=padding) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, kernel_size=self.window_size, stride=1, padding=padding) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + self.c1) * (sigma_x + sigma_y + self.c2)
        ssim_map = numerator / denominator.clamp_min(1e-6)
        return 1.0 - ssim_map.mean()


class EdgeLoss(nn.Module):
    """Edge consistency loss using fixed Sobel filters."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel_x = self.sobel_x.to(device=x.device, dtype=x.dtype).expand(channels, 1, 3, 3)
        kernel_y = self.sobel_y.to(device=x.device, dtype=x.dtype).expand(channels, 1, 3, 3)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self._gradient_magnitude(pred)
        target_edges = self._gradient_magnitude(target)
        return F.l1_loss(pred_edges, target_edges)


@dataclass
class BasicLossWeights:
    l1: float = 1.0
    ssim: float = 1.0
    edge: float = 0.5


class BasicEnhancementLoss(nn.Module):
    def __init__(self, weights: BasicLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or BasicLossWeights()
        self.l1_loss = L1ReconstructionLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1_value = self.l1_loss(pred, target)
        ssim_value = self.ssim_loss(pred, target)
        edge_value = self.edge_loss(pred, target)

        total = (
            self.weights.l1 * l1_value
            + self.weights.ssim * ssim_value
            + self.weights.edge * edge_value
        )

        return {
            "total": total,
            "l1": l1_value,
            "ssim": ssim_value,
            "edge": edge_value,
        }
