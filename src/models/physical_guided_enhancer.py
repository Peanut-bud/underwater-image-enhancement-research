"""Unified physical-guided enhancer."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.physics.airlight import airlight_tensor_to_broadcast

from .parameter_estimator import ParameterEstimator
from .physics_reconstruction import PhysicsReconstruction
from .refinement_net import RefinementNet


class PhysicalGuidedEnhancer(nn.Module):
    def __init__(self, base_channels: int = 32, t_min: float = 0.05, refinement_blocks: int = 2) -> None:
        super().__init__()
        self.parameter_estimator = ParameterEstimator(base_channels=base_channels)
        self.physics_reconstruction = PhysicsReconstruction(t_min=t_min)
        self.refinement_net = RefinementNet(hidden_channels=base_channels, blocks=refinement_blocks)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        airlight, transmission = self.parameter_estimator(x)
        rough = self.physics_reconstruction(x, airlight, transmission)
        broadcast_airlight = airlight_tensor_to_broadcast(airlight, spatial_size=x.shape[-2:])
        features = torch.cat([x, rough, transmission, broadcast_airlight], dim=1)
        enhanced = self.refinement_net(rough, features)
        return {
            "input": x,
            "airlight": airlight,
            "transmission": transmission,
            "rough": rough,
            "enhanced": enhanced,
        }


def build_physical_guided_enhancer(base_channels: int = 32, t_min: float = 0.05, refinement_blocks: int = 2) -> PhysicalGuidedEnhancer:
    return PhysicalGuidedEnhancer(base_channels=base_channels, t_min=t_min, refinement_blocks=refinement_blocks)
