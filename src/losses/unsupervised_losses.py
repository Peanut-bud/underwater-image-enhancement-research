"""Unsupervised loss stack for stage-4 adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


class ColorConstancyLoss(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        channel_means = image.mean(dim=(2, 3))
        rg = (channel_means[:, 0] - channel_means[:, 1]).pow(2)
        rb = (channel_means[:, 0] - channel_means[:, 2]).pow(2)
        gb = (channel_means[:, 1] - channel_means[:, 2]).pow(2)
        return (rg + rb + gb).mean()


class ExposureBalanceLoss(nn.Module):
    def __init__(self, target_mean: float = 0.5) -> None:
        super().__init__()
        self.target_mean = target_mean

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patch_mean = image.mean(dim=(2, 3))
        return (patch_mean - self.target_mean).abs().mean()


class TransmissionSmoothLoss(nn.Module):
    def forward(self, transmission: torch.Tensor) -> torch.Tensor:
        diff_x = (transmission[:, :, :, 1:] - transmission[:, :, :, :-1]).abs().mean()
        diff_y = (transmission[:, :, 1:, :] - transmission[:, :, :-1, :]).abs().mean()
        return diff_x + diff_y


class IdentityRegularizationLoss(nn.Module):
    def forward(self, enhanced: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        return (enhanced - input_image).abs().mean()


@dataclass
class AdaptationLossWeights:
    color_constancy: float = 1.0
    exposure_balance: float = 1.0
    transmission_smooth: float = 0.5
    identity_regularization: float = 0.2


class AdaptationLossStack(nn.Module):
    def __init__(self, weights: AdaptationLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or AdaptationLossWeights()
        self.color_constancy = ColorConstancyLoss()
        self.exposure_balance = ExposureBalanceLoss()
        self.transmission_smooth = TransmissionSmoothLoss()
        self.identity_regularization = IdentityRegularizationLoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enhanced = outputs["enhanced"]
        input_image = batch["input"]
        transmission = outputs["transmission"]

        color_loss = self.color_constancy(enhanced)
        exposure_loss = self.exposure_balance(enhanced)
        smooth_loss = self.transmission_smooth(transmission)
        identity_loss = self.identity_regularization(enhanced, input_image)

        total = (
            self.weights.color_constancy * color_loss
            + self.weights.exposure_balance * exposure_loss
            + self.weights.transmission_smooth * smooth_loss
            + self.weights.identity_regularization * identity_loss
        )
        return {
            "total": total,
            "color_constancy": color_loss,
            "exposure_balance": exposure_loss,
            "transmission_smooth": smooth_loss,
            "identity_regularization": identity_loss,
        }
