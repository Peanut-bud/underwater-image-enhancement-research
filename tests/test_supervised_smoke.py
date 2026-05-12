from __future__ import annotations

import unittest
from pathlib import Path

from src.train_supervised import main as train_supervised_main


class TestSupervisedSmoke(unittest.TestCase):
    def test_supervised_entrypoint_runs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = project_root / "checkpoints" / "physical_guided_supervised_latest.pth"
        train_supervised_main(["--config", "configs/train/supervised_base.yaml"])
        self.assertTrue(checkpoint_path.exists())
