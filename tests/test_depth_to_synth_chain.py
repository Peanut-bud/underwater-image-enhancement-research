from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from src.prepare_depth import main as prepare_depth_main
from src.prepare_synth import main as prepare_synth_main
from src.utils.io import read_json


class TestDepthToSynthChain(unittest.TestCase):
    def test_prepare_depth_then_precomputed_synth(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / "outputs" / f"test_depth_to_synth_chain_{uuid.uuid4().hex}"
        try:
            shutil.rmtree(root, ignore_errors=True)
            clean_root = root / "data" / "clean_source"
            clean_images = clean_root / "images"
            clean_depth = clean_root / "depth"
            raw_images = root / "data" / "raw_field" / "images"
            synthetic_root = root / "data" / "synthetic"
            split_root = root / "data" / "splits"
            depth_config = root / "depth_build.yaml"
            synth_config = root / "synth_build.yaml"
            synth_manifest = root / "data" / "raw_field" / "manifests" / "synth_build_manifest.json"
            clean_images.mkdir(parents=True, exist_ok=True)
            raw_images.mkdir(parents=True, exist_ok=True)

            for index, color in enumerate(((180, 120, 80), (90, 180, 210), (150, 160, 90)), start=1):
                Image.new("RGB", (40, 30), color=color).save(clean_images / f"scene_{index}.png")
            Image.new("RGB", (40, 30), color=(70, 90, 110)).save(raw_images / "field_01.png")

            depth_config.write_text(
                "\n".join(
                    [
                        "project:",
                        "  name: underwater_image_enhancement",
                        "  stage: stage1_depth_prepare",
                        "data:",
                        f"  clean_source_root: \"{clean_root.as_posix()}\"",
                        f"  clean_depth_root: \"{clean_depth.as_posix()}\"",
                        "  image_extensions: [\".png\", \".jpg\", \".jpeg\"]",
                        "  recursive: false",
                        "model:",
                        "  depth_backend: \"placeholder_inverse_luminance\"",
                        "training:",
                        "  enabled: false",
                        "loss:",
                        "  enabled: false",
                        "output:",
                        f"  manifest_path: \"{(clean_root / 'manifests' / 'depth_build_manifest.json').as_posix()}\"",
                        "  overwrite: true",
                        "runtime:",
                        "  dry_run: false",
                        "depth:",
                        "  image_size: [48, 48]",
                    ]
                ),
                encoding="utf-8",
            )
            prepare_depth_main(["--config", str(depth_config)])

            synth_config.write_text(
                "\n".join(
                    [
                        "project:",
                        "  name: underwater_image_enhancement",
                        "  stage: stage1_synthesis",
                        "data:",
                        f"  clean_source_root: \"{clean_root.as_posix()}\"",
                        f"  clean_depth_root: \"{clean_depth.as_posix()}\"",
                        f"  raw_field_root: \"{(root / 'data' / 'raw_field').as_posix()}\"",
                        f"  synthetic_root: \"{synthetic_root.as_posix()}\"",
                        f"  split_dir: \"{split_root.as_posix()}\"",
                        "  image_extensions: [\".png\", \".jpg\", \".jpeg\"]",
                        "  recursive: false",
                        "  output_image_format: \".png\"",
                        "  save_depth: true",
                        "model:",
                        "  depth_backend: \"precomputed_depth_map\"",
                        "  transmission_mode: \"exp_beta_depth\"",
                        "training:",
                        "  enabled: false",
                        "loss:",
                        "  enabled: false",
                        "output:",
                        f"  manifest_path: \"{synth_manifest.as_posix()}\"",
                        "  overwrite: true",
                        "runtime:",
                        "  dry_run: false",
                        "  seed: 5",
                        "synthesis:",
                        "  image_size: [48, 48]",
                        "  samples_per_image: 1",
                        "  t_min: 0.15",
                        "  split_ratio:",
                        "    train: 0.7",
                        "    val: 0.2",
                        "    test: 0.1",
                        "  beta_range: [0.35, 0.8]",
                        "  airlight_range:",
                        "    r: [0.65, 0.9]",
                        "    g: [0.75, 0.98]",
                        "    b: [0.7, 0.95]",
                        "  glare:",
                        "    enabled: true",
                        "    probability: 0.3",
                        "    strength_range: [0.03, 0.08]",
                        "    spot_count_range: [1, 2]",
                        "    sigma_ratio_range: [0.08, 0.14]",
                        "  noise:",
                        "    enabled: true",
                        "    mode: \"poisson\"",
                        "    strength_range: [0.0, 0.02]",
                        "  preset_names: [\"mild_turbid\"]",
                        "  presets:",
                        "    mild_turbid:",
                        "      beta_range: [0.35, 0.65]",
                        "      glare_probability: 0.3",
                        "      noise_strength_range: [0.0, 0.02]",
                    ]
                ),
                encoding="utf-8",
            )
            prepare_synth_main(["--config", str(synth_config)])

            manifest = read_json(synth_manifest)
            self.assertEqual(manifest["clean_images"], 3)
            self.assertEqual(len(manifest["samples"]), 3)
            self.assertEqual(manifest["samples"][0]["depth_source"], "precomputed_depth_map")
            self.assertTrue((synthetic_root / "train" / "depth").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)
