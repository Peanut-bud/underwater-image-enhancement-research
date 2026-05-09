"""Path, image save, and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_project_path(path_str: str | Path, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def list_image_files(root: Path, recursive: bool, allowed_extensions: Iterable[str] | None) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (allowed_extensions or IMAGE_EXTENSIONS)
    }

    if root.is_file():
        if root.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported image extension: {root.suffix}")
        return [root]

    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_image(image: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    image.save(path)


def build_output_paths(output_dir: Path, image_path: Path, enhanced_suffix: str, compare_suffix: str) -> tuple[Path, Path]:
    stem = image_path.stem
    enhanced_path = output_dir / f"{stem}{enhanced_suffix}{image_path.suffix}"
    compare_path = output_dir / f"{stem}{compare_suffix}{image_path.suffix}"
    return enhanced_path, compare_path


def build_compare_image(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    width = original.width + enhanced.width
    height = max(original.height, enhanced.height)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(original, (0, 0))
    canvas.paste(enhanced, (original.width, 0))
    return canvas
