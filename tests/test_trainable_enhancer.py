"""Tests for the trainable stage-one enhancement backbone."""

from __future__ import annotations

import unittest

import torch

from src.models import TrainableEnhancer, build_trainable_enhancer


class TrainableEnhancerTests(unittest.TestCase):
    def test_forward_shape_is_preserved(self) -> None:
        model = build_trainable_enhancer(base_channels=16)
        model.eval()
        x = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 3, 256, 256))

    def test_model_has_trainable_parameters(self) -> None:
        model = TrainableEnhancer(base_channels=16)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(trainable_params)
        self.assertGreater(sum(p.numel() for p in trainable_params), 0)

    def test_backward_pass_runs(self) -> None:
        model = build_trainable_enhancer(base_channels=16)
        x = torch.rand(1, 3, 128, 128)
        target = torch.rand(1, 3, 128, 128)
        output = model(x)
        loss = (output - target).abs().mean()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_highlight_map_shape_and_range(self) -> None:
        model = build_trainable_enhancer(base_channels=16)
        model.eval()
        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            features = model.shallow(x)
            highlight_map = model.compute_highlight_map(features)
        self.assertEqual(tuple(highlight_map.shape), (2, 1, 128, 128))
        self.assertGreaterEqual(float(highlight_map.min()), 0.0)
        self.assertLessEqual(float(highlight_map.max()), 1.0)

    def test_highlight_modulation_changes_features(self) -> None:
        model = build_trainable_enhancer(base_channels=16)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            features = model.shallow(x)
            highlight_map = model.compute_highlight_map(features)
            modulated = model.apply_highlight_modulation(features, highlight_map)
        self.assertEqual(tuple(modulated.shape), tuple(features.shape))
        self.assertFalse(torch.equal(features, modulated))


if __name__ == "__main__":
    unittest.main()
