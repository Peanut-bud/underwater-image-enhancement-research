"""Synthetic quadruplet dataset for stage-3 supervised training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.transforms import pil_to_grayscale_tensor, pil_to_tensor
from src.utils.io import IMAGE_EXTENSIONS, read_json


class SyntheticQuadDataset(Dataset):
    """Read synthetic `(input, target, transmission, airlight)` tuples."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.split_root = self.root / split
        self.input_dir = self.split_root / "input"
        self.target_dir = self.split_root / "target"
        self.transmission_dir = self.split_root / "transmission"
        self.airlight_dir = self.split_root / "airlight"
        self.metadata_dir = self.split_root / "metadata"

        for directory in (
            self.input_dir,
            self.target_dir,
            self.transmission_dir,
            self.airlight_dir,
            self.metadata_dir,
        ):
            if not directory.exists():
                raise FileNotFoundError(f"Required synthetic directory does not exist: {directory}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No synthetic samples found under {self.split_root}")

    def _collect_samples(self) -> list[dict[str, Path]]:
        input_files = sorted(
            path for path in self.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples: list[dict[str, Path]] = []
        for input_path in input_files:
            stem = input_path.stem
            target_path = self.target_dir / input_path.name
            transmission_path = self.transmission_dir / input_path.name
            airlight_path = self.airlight_dir / f"{stem}.json"
            metadata_path = self.metadata_dir / f"{stem}.json"

            missing = [
                str(path)
                for path in (target_path, transmission_path, airlight_path, metadata_path)
                if not path.exists()
            ]
            if missing:
                preview = ", ".join(missing)
                raise FileNotFoundError(f"Synthetic sample '{stem}' is incomplete: {preview}")

            samples.append(
                {
                    "input": input_path,
                    "target": target_path,
                    "transmission": transmission_path,
                    "airlight": airlight_path,
                    "metadata": metadata_path,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_rgb(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize(self.image_size, Image.BICUBIC)

    def _resize_gray(self, image: Image.Image) -> Image.Image:
        return image.convert("L").resize(self.image_size, Image.BICUBIC)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        stem = sample["input"].stem
        input_image = self._resize_rgb(Image.open(sample["input"]))
        target_image = self._resize_rgb(Image.open(sample["target"]))
        transmission_image = self._resize_gray(Image.open(sample["transmission"]))
        airlight_payload = read_json(sample["airlight"])
        metadata_payload = read_json(sample["metadata"])

        airlight_values = airlight_payload.get("A", metadata_payload.get("A", [0.8, 0.9, 0.9]))
        if len(airlight_values) != 3:
            raise ValueError(f"Airlight for '{stem}' must contain 3 values.")

        airlight_image = Image.new(
            "RGB",
            (1, 1),
            color=tuple(int(max(min(value, 1.0), 0.0) * 255.0) for value in airlight_values),
        )
        return {
            "input": pil_to_tensor(input_image),
            "target": pil_to_tensor(target_image),
            "transmission": pil_to_grayscale_tensor(transmission_image),
            "airlight": pil_to_tensor(airlight_image),
            "metadata": metadata_payload,
            "meta": {
                "split": self.split,
                "stem": stem,
                "input_path": str(sample["input"]),
            },
        }
