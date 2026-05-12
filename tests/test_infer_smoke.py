from __future__ import annotations

import unittest
from pathlib import Path

from src.infer import main as infer_main


class TestInferSmoke(unittest.TestCase):
    def test_infer_entrypoint_saves_outputs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "outputs" / "infer"
        infer_main(["--config", "configs/infer/infer_base.yaml"])
        enhanced = sorted(output_dir.glob("*_enhanced.*"))
        compare = sorted(output_dir.glob("*_compare.*"))
        self.assertGreaterEqual(len(enhanced), 1)
        self.assertGreaterEqual(len(compare), 1)
