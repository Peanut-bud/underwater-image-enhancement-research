"""Physics reconstruction layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsReconstruction(nn.Module):
    def __init__(self, t_min: float = 0.05) -> None:
        super().__init__()
        self.t_min = t_min

    def forward(self, input_image: torch.Tensor, airlight: torch.Tensor, transmission: torch.Tensor) -> torch.Tensor:
        if airlight.ndim != 4 or airlight.shape[2:] != (1, 1):
            raise ValueError("Airlight must have shape [B, 3, 1, 1].")
        if transmission.ndim != 4 or transmission.shape[1] != 1:
            raise ValueError("Transmission must have shape [B, 1, H, W].")
        safe_transmission = transmission.clamp_min(self.t_min)
        rough = (input_image - airlight) / safe_transmission + airlight
        return rough.clamp(0.0, 1.0)
