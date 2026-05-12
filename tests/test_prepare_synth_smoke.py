from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from src.prepare_synth import main as prepare_synth_main
from src.utils.io import read_json


class TestPrepareSynthSmoke(unittest.TestCase):
    def test_prepare_synth_generates_synthetic_layout(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / "outputs" / f"test_prepare_synth_smoke_{uuid.uuid4().hex}"
        try:
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)
            clean_images = root / "data" / "clean_source" / "images"
            raw_images = root / "data" / "raw_field" / "images"
            synthetic_root = root / "data" / "synthetic"
            split_root = root / "data" / "splits"
            manifest_path = root / "data" / "raw_field" / "manifests" / "synth_build_manifest.json"
            config_path = root / "synth_build.yaml"

            clean_images.mkdir(parents=True, exist_ok=True)
            raw_images.mkdir(parents=True, exist_ok=True)

            for index, color in enumerate(((220, 90, 90), (90, 220, 120), (80, 100, 220)), start=1):
                Image.new("RGB", (40, 40), color=color).save(clean_images / f"sample_{index}.png")
            Image.new("RGB", (40, 40), color=(60, 80, 100)).save(raw_images / "field_01.png")

            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  name: underwater_image_enhancement",
                        "  stage: stage1_synthesis",
                        "data:",
                        f"  clean_source_root: \"{(root / 'data' / 'clean_source').as_posix()}\"",
                        f"  raw_field_root: \"{(root / 'data' / 'raw_field').as_posix()}\"",
                        f"  synthetic_root: \"{synthetic_root.as_posix()}\"",
                        f"  split_dir: \"{split_root.as_posix()}\"",
                        "  image_extensions: [\".png\", \".jpg\", \".jpeg\"]",
                        "  recursive: false",
                        "  output_image_format: \".png\"",
                        "model:",
                        "  depth_backend: \"placeholder_inverse_luminance\"",
                        "  transmission_mode: \"exp_beta_depth\"",
                        "training:",
                        "  enabled: false",
                        "loss:",
                        "  enabled: false",
                        "output:",
                        f"  manifest_path: \"{manifest_path.as_posix()}\"",
                        "  overwrite: true",
                        "runtime:",
                        "  dry_run: false",
                        "  seed: 7",
                        "synthesis:",
                        "  image_size: [32, 32]",
                        "  samples_per_image: 2",
                        "  t_min: 0.15",
                        "  split_ratio:",
                        "    train: 0.7",
                        "    val: 0.2",
                        "    test: 0.1",
                        "  beta_range: [0.35, 1.2]",
                        "  airlight_range:",
                        "    r: [0.65, 0.9]",
                        "    g: [0.75, 0.98]",
                        "    b: [0.7, 0.95]",
                        "  depth:",
                        "    normalize: true",
                        "  glare:",
                        "    enabled: true",
                        "    probability: 0.5",
                        "    strength_range: [0.03, 0.12]",
                        "    spot_count_range: [1, 2]",
                        "    sigma_ratio_range: [0.08, 0.16]",
                        "  noise:",
                        "    enabled: true",
                        "    mode: \"poisson\"",
                        "    strength_range: [0.0, 0.03]",
                        "  preset_names: [\"mild_turbid\", \"medium_turbid\"]",
                        "  presets:",
                        "    mild_turbid:",
                        "      beta_range: [0.35, 0.65]",
                        "      glare_probability: 0.25",
                        "      noise_strength_range: [0.0, 0.015]",
                        "    medium_turbid:",
                        "      beta_range: [0.65, 0.9]",
                        "      glare_probability: 0.5",
                        "      noise_strength_range: [0.01, 0.025]",
                    ]
                ),
                encoding="utf-8",
            )

            prepare_synth_main(["--config", str(config_path)])

            generated_inputs = sorted((synthetic_root / "train" / "input").glob("*.png"))
            generated_targets = sorted((synthetic_root / "train" / "target").glob("*.png"))
            generated_depth = sorted((synthetic_root / "train" / "depth").glob("*.png"))
            generated_meta = sorted((synthetic_root / "train" / "metadata").glob("*.json"))
            self.assertGreaterEqual(len(generated_inputs), 1)
            self.assertEqual(len(generated_inputs), len(generated_targets))
            self.assertEqual(len(generated_inputs), len(generated_depth))
            self.assertEqual(len(generated_inputs), len(generated_meta))

            self.assertTrue((split_root / "train.txt").exists())
            self.assertTrue((split_root / "val.txt").exists())
            self.assertTrue((split_root / "test.txt").exists())
            self.assertTrue(manifest_path.exists())

            manifest = read_json(manifest_path)
            self.assertEqual(manifest["clean_images"], 3)
            self.assertEqual(manifest["raw_images"], 1)
            self.assertEqual(manifest["samples_per_image"], 2)
            self.assertEqual(len(manifest["samples"]), 6)
            self.assertIn("generated_at_utc", manifest)
            self.assertIn("source_manifests", manifest)
            self.assertIn("depth_image", manifest["samples"][0])
            self.assertTrue((root / "data" / "clean_source" / "manifests" / "clean_source_index.json").exists())
            self.assertTrue((root / "data" / "raw_field" / "manifests" / "raw_field_index.json").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)
