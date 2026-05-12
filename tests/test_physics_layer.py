from __future__ import annotations

import unittest

import torch

from src.models.physics_reconstruction import PhysicsReconstruction


class TestPhysicsLayer(unittest.TestCase):
    def test_reconstruction_clamps_transmission(self) -> None:
        layer = PhysicsReconstruction(t_min=0.1)
        input_image = torch.full((1, 3, 4, 4), 0.5)
        airlight = torch.full((1, 3, 1, 1), 0.8)
        transmission = torch.zeros((1, 1, 4, 4))
        rough = layer(input_image, airlight, transmission)
        self.assertEqual(rough.shape, (1, 3, 4, 4))
        self.assertTrue(torch.all(rough >= 0.0))
        self.assertTrue(torch.all(rough <= 1.0))
