from __future__ import annotations

import unittest
from pathlib import Path

from src.datasets import SyntheticQuadDataset


class TestSyntheticQuadDataset(unittest.TestCase):
    def test_reads_quadruplet_fields(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        dataset = SyntheticQuadDataset(project_root / "data" / "synthetic", split="train", image_size=(64, 64))
        sample = dataset[0]
        self.assertIn("input", sample)
        self.assertIn("target", sample)
        self.assertIn("transmission", sample)
        self.assertIn("airlight", sample)
        self.assertEqual(sample["input"].shape, (3, 64, 64))
        self.assertEqual(sample["target"].shape, (3, 64, 64))
        self.assertEqual(sample["transmission"].shape, (1, 64, 64))
        self.assertEqual(sample["airlight"].shape, (3, 1, 1))
