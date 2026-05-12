from __future__ import annotations

import random
import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from src.physics.depth_proxy import build_depth_estimator
from src.physics.synthesis import synthesize_sample


class TestSynthesisSingleSample(unittest.TestCase):
    def test_synthesize_sample_returns_quadruplet_payload(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        tmp_root = project_root / "outputs" / f"test_synthesis_single_sample_{uuid.uuid4().hex}"
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
            tmp_root.mkdir(parents=True, exist_ok=True)
            image_path = tmp_root / "clean.png"
            Image.new("RGB", (48, 32), color=(120, 180, 200)).save(image_path)

            config = {
                "model": {"depth_backend": "placeholder_inverse_luminance"},
                "runtime": {"seed": 123},
                "synthesis": {
                    "image_size": [64, 64],
                    "t_min": 0.15,
                    "beta_range": [0.4, 0.9],
                    "airlight_range": {
                        "r": [0.7, 0.8],
                        "g": [0.8, 0.9],
                        "b": [0.75, 0.85],
                    },
                    "glare": {
                        "enabled": True,
                        "probability": 1.0,
                        "strength_range": [0.05, 0.05],
                    },
                    "noise": {
                        "enabled": True,
                        "mode": "gaussian",
                        "strength_range": [0.01, 0.01],
                    },
                    "preset_names": ["medium_turbid"],
                    "presets": {
                        "medium_turbid": {
                            "beta_range": [0.4, 0.9],
                            "glare_probability": 1.0,
                            "noise_strength_range": [0.01, 0.01],
                        }
                    },
                },
            }

            depth_estimator = build_depth_estimator(config)
            sample = synthesize_sample(
                source_path=image_path,
                split="train",
                sample_index=0,
                config=config,
                depth_estimator=depth_estimator,
                rng=random.Random(123),
            )

            self.assertEqual(sample.sample_id, "clean-s01")
            self.assertEqual(sample.target.shape, (1, 3, 64, 64))
            self.assertEqual(sample.depth.shape, (1, 1, 64, 64))
            self.assertEqual(sample.transmission.shape, (1, 1, 64, 64))
            self.assertEqual(sample.input_image.shape, (1, 3, 64, 64))
            self.assertEqual(len(sample.airlight), 3)
            self.assertIn("beta", sample.metadata)
            self.assertIn("A", sample.metadata)
            self.assertGreaterEqual(sample.metadata["transmission_min"], 0.15)
            self.assertLessEqual(sample.metadata["transmission_max"], 1.0)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
