"""Supervised loss stack for stage-3 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perceptual import VGGPerceptualLoss


class SSIMLoss(nn.Module):
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
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        grad_x = F.conv2d(x, self.sobel_x.expand(channels, 1, 3, 3), padding=1, groups=channels)
        grad_y = F.conv2d(x, self.sobel_y.expand(channels, 1, 3, 3), padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient(pred), self._gradient(target))


@dataclass
class SupervisedLossWeights:
    recon: float = 1.0
    ssim: float = 0.5
    edge: float = 0.2
    transmission: float = 0.5
    airlight: float = 0.2
    perceptual: float = 0.1


class SupervisedLossStack(nn.Module):
    def __init__(self, weights: SupervisedLossWeights | None = None, use_stub_perceptual: bool = True) -> None:
        super().__init__()
        self.weights = weights or SupervisedLossWeights()
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.perceptual = VGGPerceptualLoss(use_stub_backbone=use_stub_perceptual)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enhanced = outputs["enhanced"]
        target = batch["target"]
        transmission_pred = outputs["transmission"]
        transmission_gt = batch["transmission"]
        airlight_pred = outputs["airlight"]
        airlight_gt = batch["airlight"]

        recon_loss = F.l1_loss(enhanced, target)
        ssim_loss = self.ssim(enhanced, target)
        edge_loss = self.edge(enhanced, target)
        transmission_loss = F.mse_loss(transmission_pred, transmission_gt)
        airlight_loss = F.mse_loss(airlight_pred, airlight_gt)
        perceptual_loss = self.perceptual(enhanced, target)

        total = (
            self.weights.recon * recon_loss
            + self.weights.ssim * ssim_loss
            + self.weights.edge * edge_loss
            + self.weights.transmission * transmission_loss
            + self.weights.airlight * airlight_loss
            + self.weights.perceptual * perceptual_loss
        )
        return {
            "total": total,
            "recon": recon_loss,
            "ssim": ssim_loss,
            "edge": edge_loss,
            "transmission": transmission_loss,
            "airlight": airlight_loss,
            "perceptual": perceptual_loss,
        }
