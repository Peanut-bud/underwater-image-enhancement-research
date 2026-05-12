"""Stage-1 synthesis scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.io import list_image_files


@dataclass
class SynthesisSummary:
    clean_images: int
    raw_images: int
    synthetic_train: int
    synthetic_val: int
    synthetic_test: int


def build_synthesis_summary(clean_root: Path, raw_root: Path, synthetic_root: Path) -> SynthesisSummary:
    clean_images = len(list_image_files(clean_root / "images", recursive=False, allowed_extensions=None))
    raw_images = len(list_image_files(raw_root / "images", recursive=False, allowed_extensions=None))

    def _count(split: str) -> int:
        split_dir = synthetic_root / split / "input"
        if not split_dir.exists():
            return 0
        return len(list_image_files(split_dir, recursive=False, allowed_extensions=None))

    return SynthesisSummary(
        clean_images=clean_images,
        raw_images=raw_images,
        synthetic_train=_count("train"),
        synthetic_val=_count("val"),
        synthetic_test=_count("test"),
    )
