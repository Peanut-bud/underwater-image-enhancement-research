"""Minimal paired dataset for stage-one training smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.transforms import build_paired_transform
from src.utils.io import IMAGE_EXTENSIONS


class PairedImageDataset(Dataset):
    """Read same-name input/target image pairs from a split directory.

    Expected structure:

    data/
      train/
        input/
        target/
      val/
        input/
        target/
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (512, 512),
        patch_training: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.input_dir = self.root / split / "input"
        self.target_dir = self.root / split / "target"
        self.transform = build_paired_transform(
            image_size=image_size,
            split=split,
            patch_training=patch_training,
        )

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory does not exist: {self.target_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No paired images found in split '{split}' under {self.root}")

    def _collect_samples(self) -> list[dict[str, Path]]:
        input_files = sorted(
            p for p in self.input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        target_lookup = {
            p.name: p for p in self.target_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        samples: list[dict[str, Path]] = []
        missing_targets: list[str] = []
        for input_path in input_files:
            target_path = target_lookup.get(input_path.name)
            if target_path is None:
                missing_targets.append(input_path.name)
                continue
            samples.append({"input": input_path, "target": target_path})

        if missing_targets:
            preview = ", ".join(missing_targets[:5])
            raise FileNotFoundError(
                f"Missing target images for split '{self.split}'. "
                f"Examples: {preview}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        input_image = Image.open(sample["input"]).convert("RGB")
        target_image = Image.open(sample["target"]).convert("RGB")
        input_tensor, target_tensor = self.transform(input_image, target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "meta": {
                "split": self.split,
                "input_path": str(sample["input"]),
                "target_path": str(sample["target"]),
                "filename": sample["input"].name,
            },
        }
