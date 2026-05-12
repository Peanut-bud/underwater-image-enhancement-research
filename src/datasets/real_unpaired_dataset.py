"""Real unpaired dataset for stage-4 adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.transforms import pil_to_tensor
from src.utils.io import IMAGE_EXTENSIONS


class RealUnpairedDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.image_dir = self.root / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Real unpaired split does not exist: {self.image_dir}")

        self.samples = sorted(
            path for path in self.image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.samples:
            raise ValueError(f"No real unpaired samples found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize(self.image_size, Image.BICUBIC)
        return {
            "input": pil_to_tensor(image),
            "meta": {
                "split": self.split,
                "input_path": str(image_path),
                "filename": image_path.name,
            },
        }
