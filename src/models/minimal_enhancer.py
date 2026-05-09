"""A deterministic stage-one enhancement model.

The model stays intentionally lightweight and checkpoint-optional so the
project can run inference before training infrastructure exists. It mirrors
the planned stage-one shape:

input -> shallow features -> frequency enhancement -> differential detail -> reconstruction
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


class ShallowFeatureExtractor(nn.Module):
    """Mild local contrast enhancement while preserving identity behavior."""

    def __init__(self, local_gain: float = 0.10) -> None:
        super().__init__()
        self.local_gain = local_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        contrast = x - local_mean
        return torch.clamp(x + self.local_gain * contrast, 0.0, 1.0)


class FrequencyEnhancementBlock(nn.Module):
    """Simple color balancing plus high-frequency reinforcement.

    This is not a full dual-domain paper implementation. It is a stable,
    dependency-free approximation that nudges the image toward a more balanced
    color distribution and restores a modest amount of high-frequency detail.
    """

    def __init__(self, color_strength: float = 0.18, high_freq_gain: float = 0.22) -> None:
        super().__init__()
        self.color_strength = color_strength
        self.high_freq_gain = high_freq_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gray-world style channel balancing to reduce underwater green/yellow cast.
        channel_mean = x.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        target_mean = channel_mean.mean(dim=1, keepdim=True)
        gain = (target_mean / channel_mean).clamp(0.75, 1.25)
        balanced = torch.clamp(x * (1.0 - self.color_strength + self.color_strength * gain), 0.0, 1.0)

        # High-frequency reinforcement with a larger blur than the shallow stage.
        low_freq = F.avg_pool2d(balanced, kernel_size=5, stride=1, padding=2)
        high_freq = balanced - low_freq
        enhanced = balanced + self.high_freq_gain * high_freq
        return torch.clamp(enhanced, 0.0, 1.0)


class DifferentialConvolutionDetailBlock(nn.Module):
    """Multi-branch fixed-kernel detail enhancement.

    The branches approximate the later differential-convolution family without
    requiring trainable weights in stage one.
    """

    def __init__(self, detail_gain: float = 0.10) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],  # center / laplacian
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],  # horizontal gradient
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],  # vertical gradient
            ],
            dtype=torch.float32,
        )
        self.register_buffer("kernels", kernels.unsqueeze(1))
        self.detail_gain = detail_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        branch_responses = []
        for kernel in self.kernels:
            expanded = kernel.unsqueeze(0).expand(channels, 1, 3, 3)
            response = F.conv2d(x, expanded, padding=1, groups=channels)
            branch_responses.append(response)

        detail = torch.stack(branch_responses, dim=0).mean(dim=0)
        enhanced = x + self.detail_gain * detail
        return torch.clamp(enhanced, 0.0, 1.0)


class ReconstructionHead(nn.Module):
    """Residual reconstruction with luminance-aware gating.

    Highlights receive a smaller residual gain to avoid blowing out already
    bright regions while darker areas benefit more from enhancement.
    """

    def __init__(self, residual_gain: float = 0.85, highlight_protection: float = 0.65) -> None:
        super().__init__()
        self.residual_gain = residual_gain
        self.highlight_protection = highlight_protection

    def forward(self, original: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        residual = features - original
        luminance = _rgb_to_luminance(original)
        gate = 1.0 - self.highlight_protection * luminance
        reconstructed = original + self.residual_gain * gate * residual
        return torch.clamp(reconstructed, 0.0, 1.0)


class MinimalEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shallow = ShallowFeatureExtractor()
        self.frequency = FrequencyEnhancementBlock()
        self.detail = DifferentialConvolutionDetailBlock()
        self.reconstruction = ReconstructionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shallow(x)
        features = self.frequency(features)
        features = self.detail(features)
        return self.reconstruction(x, features)


def build_minimal_enhancer() -> MinimalEnhancer:
    return MinimalEnhancer()
