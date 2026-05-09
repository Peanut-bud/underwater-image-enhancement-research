"""Tests for the stage-one basic loss stack."""

from __future__ import annotations

import unittest

import torch

from src.losses import BasicEnhancementLoss, EdgeLoss, L1ReconstructionLoss, SSIMLoss


class BasicLossTests(unittest.TestCase):
    def test_individual_losses_run(self) -> None:
        pred = torch.rand(2, 3, 128, 128)
        target = torch.rand(2, 3, 128, 128)

        l1_value = L1ReconstructionLoss()(pred, target)
        ssim_value = SSIMLoss()(pred, target)
        edge_value = EdgeLoss()(pred, target)

        self.assertTrue(torch.isfinite(l1_value))
        self.assertTrue(torch.isfinite(ssim_value))
        self.assertTrue(torch.isfinite(edge_value))

    def test_combined_loss_returns_expected_keys(self) -> None:
        pred = torch.rand(2, 3, 128, 128)
        target = torch.rand(2, 3, 128, 128)

        losses = BasicEnhancementLoss()(pred, target)
        self.assertEqual(set(losses.keys()), {"total", "l1", "ssim", "edge"})
        for value in losses.values():
            self.assertTrue(torch.isfinite(value))

    def test_combined_loss_supports_backward(self) -> None:
        pred = torch.rand(1, 3, 64, 64, requires_grad=True)
        target = torch.rand(1, 3, 64, 64)

        losses = BasicEnhancementLoss()(pred, target)
        losses["total"].backward()
        self.assertIsNotNone(pred.grad)


if __name__ == "__main__":
    unittest.main()
