"""Airlight utilities."""

from __future__ import annotations

import random

import torch


def clamp_airlight(airlight: torch.Tensor) -> torch.Tensor:
    return airlight.clamp(0.0, 1.0)


def sample_airlight(rng: random.Random, airlight_cfg: dict) -> list[float]:
    """Sample a global RGB airlight vector."""

    def _sample(channel: str, default_range: tuple[float, float]) -> float:
        values = airlight_cfg.get(channel, list(default_range))
        low, high = float(values[0]), float(values[1])
        return rng.uniform(low, high)

    return [
        _sample("r", (0.65, 0.9)),
        _sample("g", (0.75, 0.98)),
        _sample("b", (0.7, 0.95)),
    ]


def airlight_to_tensor(values: list[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if len(values) != 3:
        raise ValueError("Airlight values must contain exactly 3 channels.")
    airlight = torch.tensor(values, device=device, dtype=dtype).view(1, 3, 1, 1)
    return clamp_airlight(airlight)


def airlight_tensor_to_broadcast(airlight: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
    if airlight.ndim != 4 or airlight.shape[2:] != (1, 1):
        raise ValueError("Airlight tensor must have shape [B, 3, 1, 1].")
    height, width = spatial_size
    return airlight.expand(-1, -1, height, width)
