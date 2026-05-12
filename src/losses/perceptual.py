"""Perceptual loss with a lightweight fallback implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
    def __init__(self, use_stub_backbone: bool = True) -> None:
        super().__init__()
        self.use_stub_backbone = use_stub_backbone

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._stub_forward(pred, target)

    def _stub_forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = F.l1_loss(pred, target)
        total = total + F.l1_loss(F.avg_pool2d(pred, 2), F.avg_pool2d(target, 2))
        total = total + F.l1_loss(F.avg_pool2d(pred, 4), F.avg_pool2d(target, 4))
        return total / 3.0
