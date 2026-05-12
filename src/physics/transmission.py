"""Transmission map helpers."""

from __future__ import annotations

import torch


def transmission_from_depth(depth: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    beta_tensor = beta if isinstance(beta, torch.Tensor) else torch.tensor(beta, dtype=depth.dtype, device=depth.device)
    transmission = torch.exp(-beta_tensor * depth.clamp_min(0.0))
    return transmission.clamp(0.0, 1.0)
