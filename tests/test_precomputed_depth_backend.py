from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from src.physics.depth_proxy import build_depth_estimator
from src.preprocessing.transforms import pil_to_tensor


class TestPrecomputedDepthBackend(unittest.TestCase):
    def test_precomputed_depth_map_backend_reads_matching_depth(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / "outputs" / f"test_precomputed_depth_backend_{uuid.uuid4().hex}"
        try:
            shutil.rmtree(root, ignore_errors=True)
            clean_root = root / "data" / "clean_source"
            image_root = clean_root / "images"
            depth_root = clean_root / "depth"
            image_root.mkdir(parents=True, exist_ok=True)
            depth_root.mkdir(parents=True, exist_ok=True)

            image_path = image_root / "sample_a.png"
            depth_path = depth_root / "sample_a.npy"
            Image.new("RGB", (32, 24), color=(120, 150, 180)).save(image_path)
            depth_values = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (24, 1))
            np.save(depth_path, depth_values)

            image_tensor = pil_to_tensor(Image.open(image_path).convert("RGB").resize((64, 64), Image.BICUBIC)).unsqueeze(0)
            config = {
                "data": {
                    "clean_depth_root": str(depth_root),
                },
                "model": {
                    "depth_backend": "precomputed_depth_map",
                },
            }

            estimator = build_depth_estimator(config, project_root=project_root)
            depth = estimator(image_tensor, image_path)

            self.assertEqual(depth.shape, (1, 1, 64, 64))
            self.assertGreaterEqual(float(depth.min().item()), 0.0)
            self.assertLessEqual(float(depth.max().item()), 1.0)
            left_value = float(depth[0, 0, 32, 0].item())
            right_value = float(depth[0, 0, 32, -1].item())
            self.assertLess(left_value, right_value)
        finally:
            shutil.rmtree(root, ignore_errors=True)
