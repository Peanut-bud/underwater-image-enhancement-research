"""Stage-1 depth preparation entrypoint."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.physics.depth_proxy import build_depth_estimator
from src.preprocessing.transforms import pil_to_tensor
from src.utils.config import load_config
from src.utils.io import ensure_dir, list_image_files, write_json, resolve_project_path


@dataclass
class DepthBuildSummary:
    input_images: int
    generated_depth_maps: int
    depth_root: str
    manifest_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean-source depth maps for stage-1 synthesis.")
    parser.add_argument("--config", default="configs/data/depth_build.yaml", help="Path to the depth-build config.")
    return parser.parse_args(argv)


def _handle_remove_error(func, target, exc_info) -> None:
    try:
        os.chmod(target, 0o700)
    except OSError:
        pass
    func(target)


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_handle_remove_error)


def _tensor_to_gray_image(depth_tensor) -> Image.Image:
    clipped = depth_tensor.detach().cpu().clamp(0.0, 1.0)
    if clipped.ndim == 4:
        clipped = clipped[0]
    array = (clipped.squeeze(0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


def run_depth_pipeline(config: dict, project_root: Path) -> DepthBuildSummary:
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    runtime_cfg = config.get("runtime", {})
    depth_cfg = config.get("depth", {})

    clean_root = resolve_project_path(data_cfg["clean_source_root"], project_root)
    depth_root = resolve_project_path(data_cfg.get("clean_depth_root", "./data/clean_source/depth"), project_root)
    manifest_path = resolve_project_path(output_cfg["manifest_path"], project_root)
    image_paths = list_image_files(
        clean_root / "images",
        recursive=bool(data_cfg.get("recursive", False)),
        allowed_extensions=data_cfg.get("image_extensions"),
    )

    if bool(runtime_cfg.get("dry_run", False)):
        return DepthBuildSummary(
            input_images=len(image_paths),
            generated_depth_maps=len(image_paths),
            depth_root=str(depth_root),
            manifest_path=str(manifest_path),
        )

    if bool(output_cfg.get("overwrite", False)):
        _clear_directory(depth_root)

    ensure_dir(depth_root)
    estimator = build_depth_estimator(config, project_root=project_root)
    manifest_samples: list[dict] = []
    resize_values = depth_cfg.get("image_size")
    image_size = tuple(int(value) for value in resize_values) if resize_values else None

    for source_path in image_paths:
        image = Image.open(source_path).convert("RGB")
        if image_size is not None:
            image = image.resize(image_size, Image.BICUBIC)

        image_tensor = pil_to_tensor(image).unsqueeze(0)
        depth = estimator(image_tensor, source_path)
        depth_image = _tensor_to_gray_image(depth)
        depth_path = depth_root / f"{source_path.stem}.png"
        depth_image.save(depth_path)

        manifest_samples.append(
            {
                "source_image": source_path.as_posix(),
                "depth_image": depth_path.as_posix(),
                "depth_source": str(config.get("model", {}).get("depth_backend", "placeholder_inverse_luminance")),
                "image_size": [depth_image.width, depth_image.height],
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": config.get("project", {}),
        "config_snapshot": config,
        "input_images": len(image_paths),
        "generated_depth_maps": len(manifest_samples),
        "depth_root": depth_root.as_posix(),
        "samples": manifest_samples,
    }
    write_json(manifest_path, payload)
    return DepthBuildSummary(
        input_images=len(image_paths),
        generated_depth_maps=len(manifest_samples),
        depth_root=str(depth_root),
        manifest_path=str(manifest_path),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_project_path(args.config, project_root))
    summary = run_depth_pipeline(config, project_root)
    print("[INFO] Stage-1 depth preparation complete.")
    print(f"[INFO] Input images : {summary.input_images}")
    print(f"[INFO] Depth maps   : {summary.generated_depth_maps}")
    print(f"[INFO] Depth root   : {summary.depth_root}")
    print(f"[INFO] Manifest     : {summary.manifest_path}")


if __name__ == "__main__":
    main()
