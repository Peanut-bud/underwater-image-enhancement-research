"""Depth backends for stage-1 synthesis."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np
from PIL import Image

import torch

from src.preprocessing.transforms import pil_to_grayscale_tensor
from src.utils.io import resolve_project_path


def _append_vendor_python(project_root: Path | None) -> None:
    if project_root is None:
        return
    vendor_root = (project_root / ".vendor" / "python").resolve()
    if vendor_root.exists():
        vendor_str = str(vendor_root)
        if vendor_str not in sys.path:
            sys.path.append(vendor_str)


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """Normalize depth per-sample into ``[0, 1]``."""

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("Depth normalization expects shape [B, 1, H, W].")

    flattened = depth.flatten(start_dim=2)
    min_values = flattened.min(dim=2).values.view(-1, 1, 1, 1)
    max_values = flattened.max(dim=2).values.view(-1, 1, 1, 1)
    denom = (max_values - min_values).clamp_min(1e-6)
    return ((depth - min_values) / denom).clamp(0.0, 1.0)


def estimate_depth_proxy(image: torch.Tensor) -> torch.Tensor:
    """Estimate a lightweight depth proxy from luminance."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Depth proxy expects input shaped as [B, 3, H, W].")

    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    depth = 1.0 - luminance.clamp(0.0, 1.0)
    return normalize_depth(depth)


def _tensor_to_pil_rgb(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Expected RGB tensor shaped as [1, 3, H, W].")
    array = image[0].detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _depth_output_to_tensor(depth_output, image_size: tuple[int, int]) -> torch.Tensor:
    if isinstance(depth_output, Image.Image):
        image = depth_output.convert("L")
    elif isinstance(depth_output, np.ndarray):
        image = Image.fromarray(depth_output.astype(np.float32))
    else:
        raise TypeError(f"Unsupported depth output type: {type(depth_output)!r}")
    resized = image.resize(image_size, Image.BICUBIC)
    return normalize_depth(pil_to_grayscale_tensor(resized).unsqueeze(0))


def _find_depth_map(source_path: Path, depth_root: Path) -> Path:
    candidates = [
        depth_root / f"{source_path.stem}.png",
        depth_root / f"{source_path.stem}.jpg",
        depth_root / f"{source_path.stem}.jpeg",
        depth_root / f"{source_path.stem}.bmp",
        depth_root / f"{source_path.stem}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No precomputed depth map found for {source_path.name} under {depth_root}")


def load_precomputed_depth_map(source_path: Path, depth_root: Path, image_size: tuple[int, int]) -> torch.Tensor:
    depth_path = _find_depth_map(source_path, depth_root)
    if depth_path.suffix.lower() == ".npy":
        array = np.load(depth_path).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"Depth map must be 2D: {depth_path}")
        image = Image.fromarray(array)
    else:
        image = Image.open(depth_path).convert("L")

    resized = image.resize(image_size, Image.BICUBIC)
    depth = pil_to_grayscale_tensor(resized).unsqueeze(0)
    return normalize_depth(depth)


def build_depth_estimator(
    config: dict,
    project_root: Path | None = None,
) -> Callable[[torch.Tensor, Path | None], torch.Tensor]:
    """Build a depth estimator backend from config."""

    backend = str(config.get("model", {}).get("depth_backend", "placeholder_inverse_luminance")).strip().lower()
    if backend in {"placeholder_inverse_luminance", "inverse_luminance", "proxy"}:
        return lambda image, source_path=None: estimate_depth_proxy(image)
    if backend in {"precomputed_depth_map", "precomputed", "depth_map"}:
        if project_root is None:
            raise ValueError("project_root is required for the precomputed_depth_map backend.")
        depth_root = resolve_project_path(config.get("data", {}).get("clean_depth_root", "./data/clean_source/depth"), project_root)

        def _estimate_from_depth_map(image: torch.Tensor, source_path: Path | None = None) -> torch.Tensor:
            if source_path is None:
                raise ValueError("source_path is required for the precomputed_depth_map backend.")
            spatial_size = (int(image.shape[3]), int(image.shape[2]))
            return load_precomputed_depth_map(source_path, depth_root, spatial_size)

        return _estimate_from_depth_map
    if backend in {"depth_anything_v1_hf", "depth_anything_hf", "depth_anything_v2_hf"}:
        _append_vendor_python(project_root)
        try:
            from transformers import pipeline  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The selected Depth Anything backend requires `transformers`. "
                "Install it in the project environment or the repo-local `.vendor/python` directory before using this backend."
            ) from exc

        model_id = str(
            config.get("model", {}).get(
                "model_id",
                "LiheYoung/depth-anything-small-hf"
                if backend != "depth_anything_v2_hf"
                else "depth-anything/Depth-Anything-V2-Small-hf",
            )
        )
        depth_pipeline = pipeline(task="depth-estimation", model=model_id)

        def _estimate_with_hf(image: torch.Tensor, source_path: Path | None = None) -> torch.Tensor:
            del source_path
            pil_image = _tensor_to_pil_rgb(image)
            result = depth_pipeline(pil_image)
            if "depth" not in result:
                raise ValueError("Depth Anything pipeline output does not contain a `depth` field.")
            spatial_size = (int(image.shape[3]), int(image.shape[2]))
            return _depth_output_to_tensor(result["depth"], spatial_size)

        return _estimate_with_hf
    raise ValueError(f"Unsupported depth backend: {backend}")
