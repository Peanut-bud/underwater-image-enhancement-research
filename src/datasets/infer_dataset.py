"""Dataset wrapper for single-image or directory inference."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from src.utils.io import list_image_files


class InferImageDataset(Dataset):
    def __init__(self, root: str | Path, recursive: bool = False) -> None:
        self.root = Path(root)
        self.samples = list_image_files(self.root, recursive=recursive, allowed_extensions=None)
        if not self.samples:
            raise ValueError(f"No inference images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Path:
        return self.samples[index]
