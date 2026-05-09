"""Stage-one inference preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image


def read_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image, input_size: Tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int], Image.Image]:
    original_size = (image.width, image.height)
    resized = image.resize(input_size, Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size, resized


def postprocess_tensor(tensor: torch.Tensor, output_size: tuple[int, int]) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0).clamp(0.0, 1.0)
    array = tensor.permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255.0).round().astype(np.uint8))
    if image.size != output_size:
        image = image.resize(output_size, Image.BICUBIC)
    return image
