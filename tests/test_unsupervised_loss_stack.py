from __future__ import annotations

import unittest

import torch

from src.losses import AdaptationLossStack


class TestUnsupervisedLossStack(unittest.TestCase):
    def test_loss_stack_returns_named_terms(self) -> None:
        loss_stack = AdaptationLossStack()
        outputs = {
            "enhanced": torch.rand(1, 3, 32, 32),
            "transmission": torch.rand(1, 1, 32, 32),
        }
        batch = {"input": torch.rand(1, 3, 32, 32)}
        losses = loss_stack(outputs, batch)
        self.assertIn("total", losses)
        self.assertIn("color_constancy", losses)
