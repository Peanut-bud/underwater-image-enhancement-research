"""Inference pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from src.models import build_physical_guided_enhancer
from src.preprocessing.image_ops import postprocess_tensor, preprocess_image, read_rgb_image
from src.utils.io import build_compare_image, build_output_paths, save_image


def resolve_device(model_cfg: Dict[str, Any]) -> torch.device:
    device_name = str(model_cfg.get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model(model_cfg: Dict[str, Any], project_root: Path) -> torch.nn.Module:
    model = build_physical_guided_enhancer(
        base_channels=int(model_cfg.get("base_channels", 32)),
        t_min=float(model_cfg.get("t_min", 0.05)),
        refinement_blocks=int(model_cfg.get("refinement_blocks", 2)),
    )
    checkpoint = model_cfg.get("checkpoint")
    use_checkpoint = bool(model_cfg.get("use_checkpoint_if_available", True))
    if not checkpoint or not use_checkpoint:
        return model

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (project_root / checkpoint_path).resolve()

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    return model


def run_single_image(
    image_path: Path,
    model: torch.nn.Module,
    input_size: tuple[int, int],
    output_dir: Path,
    save_compare: bool,
    enhanced_suffix: str,
    compare_suffix: str,
    device: torch.device,
) -> dict[str, Path]:
    image = read_rgb_image(str(image_path))
    tensor, original_size, _ = preprocess_image(image, input_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        enhanced = model(tensor)["enhanced"]

    enhanced_image = postprocess_tensor(enhanced, original_size)
    enhanced_path, compare_path = build_output_paths(output_dir, image_path, enhanced_suffix, compare_suffix)
    save_image(enhanced_image, enhanced_path)

    result = {"enhanced": enhanced_path}
    if save_compare:
        compare = build_compare_image(image, enhanced_image)
        save_image(compare, compare_path)
        result["compare"] = compare_path
    return result
