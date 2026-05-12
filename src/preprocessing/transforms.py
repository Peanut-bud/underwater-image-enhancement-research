"""Minimal transform helpers for synthetic and real-image datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def pil_to_grayscale_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


@dataclass
class ResizePair:
    size: Tuple[int, int]

    def __call__(self, input_image: Image.Image, target_image: Image.Image) -> tuple[Image.Image, Image.Image]:
        return (
            input_image.resize(self.size, Image.BICUBIC),
            target_image.resize(self.size, Image.BICUBIC),
        )


@dataclass
class RandomCropPair:
    size: Tuple[int, int]

    def __call__(self, input_image: Image.Image, target_image: Image.Image) -> tuple[Image.Image, Image.Image]:
        crop_w, crop_h = self.size
        width, height = input_image.size
        if width < crop_w or height < crop_h:
            return input_image, target_image

        if width == crop_w and height == crop_h:
            return input_image, target_image

        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        return input_image.crop(box), target_image.crop(box)


class ToTensorPair:
    def __call__(self, input_image: Image.Image, target_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return pil_to_tensor(input_image), pil_to_tensor(target_image)


class ComposePair:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, input_image: Image.Image, target_image: Image.Image):
        current_input, current_target = input_image, target_image
        for transform in self.transforms:
            current_input, current_target = transform(current_input, current_target)
        return current_input, current_target


def build_paired_transform(
    image_size: Tuple[int, int] = (512, 512),
    split: str = "train",
    patch_training: bool = False,
) -> ComposePair:
    transforms = [ResizePair(image_size)]
    if split == "train" and patch_training:
        transforms.append(RandomCropPair(image_size))
    transforms.append(ToTensorPair())
    return ComposePair(transforms)
