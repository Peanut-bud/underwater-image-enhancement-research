"""Smoke tests for the stage-one inference pipeline."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.inference.pipeline import build_model, resolve_device, run_single_image
from src.utils.config import load_config
from src.utils.io import list_image_files, resolve_project_path


class StageOneSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.project_root / "configs" / "infer_base.yaml"
        cls.config = load_config(cls.config_path)

    def test_config_loads(self) -> None:
        self.assertIn("input", self.config)
        self.assertIn("model", self.config)
        self.assertIn("output", self.config)

    def test_input_images_are_discoverable(self) -> None:
        input_cfg = self.config["input"]
        image_root = resolve_project_path(input_cfg["path"], self.project_root)
        files = list_image_files(image_root, recursive=bool(input_cfg.get("recursive", False)), allowed_extensions=input_cfg.get("allowed_extensions"))
        self.assertGreaterEqual(len(files), 1)

    def test_model_forward_shape(self) -> None:
        model_cfg = self.config["model"]
        model = build_model(model_cfg, self.project_root)
        model.eval()
        x = torch.rand(1, 3, 512, 512)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (1, 3, 512, 512))

    def test_single_image_inference_saves_output(self) -> None:
        input_cfg = self.config["input"]
        output_cfg = self.config["output"]
        model_cfg = self.config["model"]

        image_root = resolve_project_path(input_cfg["path"], self.project_root)
        image_files = list_image_files(image_root, recursive=bool(input_cfg.get("recursive", False)), allowed_extensions=input_cfg.get("allowed_extensions"))
        image_path = image_files[0]

        temp_dir = self.project_root / "outputs" / "_smoke_test"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

        input_size_cfg = model_cfg.get("input_size", [512, 512])
        input_size = (int(input_size_cfg[0]), int(input_size_cfg[1]))
        device = resolve_device(model_cfg)
        model = build_model(model_cfg, self.project_root).to(device)
        model.eval()

        result = run_single_image(
            image_path=image_path,
            model=model,
            input_size=input_size,
            output_dir=temp_dir,
            save_compare=bool(output_cfg.get("save_compare", True)),
            enhanced_suffix=str(output_cfg.get("enhanced_suffix", "_enhanced")),
            compare_suffix=str(output_cfg.get("compare_suffix", "_compare")),
            device=device,
        )

        self.assertTrue(result["enhanced"].exists())
        enhanced = Image.open(result["enhanced"])
        original = Image.open(image_path)
        self.assertEqual(enhanced.size, original.size)
        enhanced.close()
        original.close()


if __name__ == "__main__":
    unittest.main()
