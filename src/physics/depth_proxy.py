"""Placeholder depth proxy for stage-1 synthesis scaffolding."""

from __future__ import annotations

import torch


def estimate_depth_proxy(image: torch.Tensor) -> torch.Tensor:
    """Estimate a lightweight depth proxy from luminance."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Depth proxy expects input shaped as [B, 3, H, W].")

    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 1.0 - luminance.clamp(0.0, 1.0)
