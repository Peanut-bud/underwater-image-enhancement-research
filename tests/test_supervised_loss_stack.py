from __future__ import annotations

import unittest

import torch

from src.losses import SupervisedLossStack


class TestSupervisedLossStack(unittest.TestCase):
    def test_loss_stack_returns_named_terms(self) -> None:
        loss_stack = SupervisedLossStack()
        outputs = {
            "enhanced": torch.rand(1, 3, 32, 32),
            "transmission": torch.rand(1, 1, 32, 32),
            "airlight": torch.rand(1, 3, 1, 1),
        }
        batch = {
            "target": torch.rand(1, 3, 32, 32),
            "transmission": torch.rand(1, 1, 32, 32),
            "airlight": torch.rand(1, 3, 1, 1),
        }
        losses = loss_stack(outputs, batch)
        self.assertIn("total", losses)
        self.assertIn("perceptual", losses)
