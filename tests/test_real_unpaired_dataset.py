from __future__ import annotations

import unittest
from pathlib import Path

from src.datasets import RealUnpairedDataset


class TestRealUnpairedDataset(unittest.TestCase):
    def test_reads_real_images(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        dataset = RealUnpairedDataset(project_root / "data" / "real_unsup", split="train", image_size=(64, 64))
        sample = dataset[0]
        self.assertIn("input", sample)
        self.assertEqual(sample["input"].shape, (3, 64, 64))
