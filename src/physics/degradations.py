"""Simple degradation helpers for stage-1 synthesis."""

from __future__ import annotations

import random

import torch


def apply_glare(
    image: torch.Tensor,
    strength: float = 0.05,
    rng: random.Random | None = None,
    spot_count: int = 1,
    sigma_ratio: float = 0.12,
) -> torch.Tensor:
    """Apply local Gaussian glare spots.

    This better matches the stage-one design note of injecting local
    high-exposure artifacts rather than globally brightening the image.
    """

    if strength <= 0:
        return image
    if image.ndim != 4:
        raise ValueError("Glare expects image shaped as [B, C, H, W].")

    generator = rng or random.Random()
    _, _, height, width = image.shape
    device = image.device
    dtype = image.dtype

    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype).view(1, 1, 1, width)
    glare_map = torch.zeros((image.shape[0], 1, height, width), device=device, dtype=dtype)

    safe_sigma = max(float(sigma_ratio), 1e-3)
    safe_spots = max(int(spot_count), 1)
    for _ in range(safe_spots):
        center_x = generator.uniform(-0.7, 0.7)
        center_y = generator.uniform(-0.7, 0.7)
        gaussian = torch.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2.0 * safe_sigma * safe_sigma))
        glare_map = glare_map + gaussian

    glare_map = glare_map / glare_map.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    glare_rgb = glare_map.expand(-1, image.shape[1], -1, -1)
    return (image + strength * glare_rgb).clamp(0.0, 1.0)


def apply_noise(image: torch.Tensor, mode: str = "gaussian", strength: float = 0.01) -> torch.Tensor:
    if strength <= 0:
        return image

    normalized_mode = mode.strip().lower()
    if normalized_mode == "gaussian":
        noise = torch.randn_like(image) * strength
        return (image + noise).clamp(0.0, 1.0)
    if normalized_mode == "poisson":
        scaled = (image.clamp(0.0, 1.0) * 255.0).clamp_min(1.0)
        noisy = torch.poisson(scaled) / 255.0
        mixed = image * (1.0 - strength) + noisy * strength
        return mixed.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported noise mode: {mode}")


def add_glare(image: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    return apply_glare(image, strength=strength)


def add_noise(image: torch.Tensor, strength: float = 0.01) -> torch.Tensor:
    return apply_noise(image, mode="gaussian", strength=strength)
