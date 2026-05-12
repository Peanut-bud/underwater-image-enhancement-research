"""Airlight utilities."""

from __future__ import annotations

import torch


def clamp_airlight(airlight: torch.Tensor) -> torch.Tensor:
    return airlight.clamp(0.0, 1.0)


def airlight_tensor_to_broadcast(airlight: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
    if airlight.ndim != 4 or airlight.shape[2:] != (1, 1):
        raise ValueError("Airlight tensor must have shape [B, 3, 1, 1].")
    height, width = spatial_size
    return airlight.expand(-1, -1, height, width)
