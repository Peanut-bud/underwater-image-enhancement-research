"""Simple degradation helpers for stage-1 scaffolding."""

from __future__ import annotations

import torch


def add_glare(image: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    glow = image.mean(dim=1, keepdim=True)
    return (image + strength * glow).clamp(0.0, 1.0)


def add_noise(image: torch.Tensor, strength: float = 0.01) -> torch.Tensor:
    if strength <= 0:
        return image
    noise = torch.randn_like(image) * strength
    return (image + noise).clamp(0.0, 1.0)
