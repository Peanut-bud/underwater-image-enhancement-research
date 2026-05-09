"""Tests for the minimal paired dataset."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.datasets import PairedImageDataset


class PairedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.data_root = cls.project_root / "data"

    def test_train_split_loads(self) -> None:
        dataset = PairedImageDataset(self.data_root, split="train", image_size=(512, 512), patch_training=False)
        self.assertEqual(len(dataset), 20)

        sample = dataset[0]
        self.assertIn("input", sample)
        self.assertIn("target", sample)
        self.assertIn("meta", sample)
        self.assertEqual(tuple(sample["input"].shape), (3, 512, 512))
        self.assertEqual(tuple(sample["target"].shape), (3, 512, 512))

    def test_val_split_loads(self) -> None:
        dataset = PairedImageDataset(self.data_root, split="val", image_size=(512, 512), patch_training=False)
        self.assertEqual(len(dataset), 6)

        sample = dataset[0]
        self.assertEqual(sample["meta"]["split"], "val")
        self.assertTrue(sample["meta"]["filename"])


if __name__ == "__main__":
    unittest.main()
