from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from src.prepare_depth import main as prepare_depth_main
from src.utils.io import read_json


class TestPrepareDepthSmoke(unittest.TestCase):
    def test_prepare_depth_generates_grayscale_depth_maps(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = project_root / "outputs" / f"test_prepare_depth_smoke_{uuid.uuid4().hex}"
        try:
            shutil.rmtree(root, ignore_errors=True)
            clean_images = root / "data" / "clean_source" / "images"
            depth_root = root / "data" / "clean_source" / "depth"
            manifest_path = root / "data" / "clean_source" / "manifests" / "depth_build_manifest.json"
            config_path = root / "depth_build.yaml"
            clean_images.mkdir(parents=True, exist_ok=True)

            for index, color in enumerate(((200, 120, 120), (120, 200, 140)), start=1):
                Image.new("RGB", (48, 32), color=color).save(clean_images / f"clean_{index}.png")

            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  name: underwater_image_enhancement",
                        "  stage: stage1_depth_prepare",
                        "data:",
                        f"  clean_source_root: \"{(root / 'data' / 'clean_source').as_posix()}\"",
                        f"  clean_depth_root: \"{depth_root.as_posix()}\"",
                        "  image_extensions: [\".png\", \".jpg\", \".jpeg\"]",
                        "  recursive: false",
                        "model:",
                        "  depth_backend: \"placeholder_inverse_luminance\"",
                        "  model_id: \"LiheYoung/depth-anything-small-hf\"",
                        "training:",
                        "  enabled: false",
                        "loss:",
                        "  enabled: false",
                        "output:",
                        f"  manifest_path: \"{manifest_path.as_posix()}\"",
                        "  overwrite: true",
                        "runtime:",
                        "  dry_run: false",
                        "depth:",
                        "  image_size: [64, 64]",
                    ]
                ),
                encoding="utf-8",
            )

            prepare_depth_main(["--config", str(config_path)])

            generated_depth = sorted(depth_root.glob("*.png"))
            self.assertEqual(len(generated_depth), 2)
            for depth_path in generated_depth:
                with Image.open(depth_path) as depth_image:
                    self.assertEqual(depth_image.mode, "L")

            manifest = read_json(manifest_path)
            self.assertEqual(manifest["input_images"], 2)
            self.assertEqual(manifest["generated_depth_maps"], 2)
            self.assertEqual(len(manifest["samples"]), 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)
