from __future__ import annotations

import unittest
from pathlib import Path

from src.train_adapt import main as train_adapt_main


class TestAdaptationSmoke(unittest.TestCase):
    def test_adaptation_entrypoint_runs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = project_root / "checkpoints" / "physical_guided_adapt_latest.pth"
        train_adapt_main(["--config", "configs/train/adapt_base.yaml"])
        self.assertTrue(checkpoint_path.exists())
