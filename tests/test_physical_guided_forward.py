from __future__ import annotations

import unittest

import torch

from src.models import build_physical_guided_enhancer


class TestPhysicalGuidedForward(unittest.TestCase):
    def test_forward_returns_expected_keys(self) -> None:
        model = build_physical_guided_enhancer(base_channels=8, t_min=0.05, refinement_blocks=1)
        outputs = model(torch.rand(1, 3, 32, 32))
        self.assertEqual(set(outputs.keys()), {"input", "airlight", "transmission", "rough", "enhanced"})
        self.assertEqual(outputs["airlight"].shape, (1, 3, 1, 1))
        self.assertEqual(outputs["transmission"].shape, (1, 1, 32, 32))
        self.assertEqual(outputs["enhanced"].shape, (1, 3, 32, 32))
