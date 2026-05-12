"""Stage-1 synthesis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.physics.airlight import airlight_tensor_to_broadcast, airlight_to_tensor, sample_airlight
from src.physics.degradations import apply_glare, apply_noise
from src.physics.depth_proxy import build_depth_estimator
from src.physics.transmission import transmission_from_depth
from src.preprocessing.transforms import pil_to_tensor
from src.utils.io import ensure_dir, list_image_files, save_image, write_json


def _tensor_to_image(tensor: torch.Tensor, mode: str = "RGB") -> Image.Image:
    clipped = tensor.detach().cpu().clamp(0.0, 1.0)
    if clipped.ndim == 4:
        clipped = clipped[0]
    if mode == "L":
        array = (clipped.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)
    array = (clipped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _clear_directory(path: Path) -> None:
    if path.exists():
        def _handle_remove_error(func, target, exc_info):
            try:
                os.chmod(target, 0o700)
            except OSError:
                pass
            func(target)

        shutil.rmtree(path, onerror=_handle_remove_error)


def _write_source_manifest(root: Path, image_paths: list[Path], project_root: Path, manifest_name: str) -> Path:
    manifest_dir = root / "manifests"
    ensure_dir(manifest_dir)
    manifest_path = manifest_dir / manifest_name
    payload = {
        "root": _relative_posix(root, project_root),
        "image_count": len(image_paths),
        "images": [_relative_posix(path, project_root) for path in image_paths],
    }
    write_json(manifest_path, payload)
    return manifest_path


@dataclass
class SynthesisSummary:
    clean_images: int
    raw_images: int
    synthetic_train: int
    synthetic_val: int
    synthetic_test: int
    generated_total: int = 0
    manifest_path: str = ""


@dataclass
class SynthesisSample:
    sample_id: str
    split: str
    target: torch.Tensor
    depth: torch.Tensor
    depth_origin_path: str | None
    transmission: torch.Tensor
    airlight: list[float]
    input_image: torch.Tensor
    metadata: dict[str, Any]


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
        generated_total=_count("train") + _count("val") + _count("test"),
    )


def build_split_plan(image_paths: list[Path], split_ratio: dict[str, Any], seed: int) -> dict[str, list[Path]]:
    shuffled = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_ratio = float(split_ratio.get("train", 0.7))
    val_ratio = float(split_ratio.get("val", 0.2))
    test_ratio = float(split_ratio.get("test", 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_count = int(total * train_ratio / ratio_sum)
    val_count = int(total * val_ratio / ratio_sum)
    if total >= 3 and val_ratio > 0 and val_count == 0:
        val_count = 1
    max_test = max(total - train_count - val_count, 0)
    test_count = min(int(total * test_ratio / ratio_sum), max_test)
    consumed = train_count + val_count + test_count
    train_count += total - consumed

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def sample_degradation_params(config: dict, rng: random.Random) -> dict[str, Any]:
    synthesis_cfg = config.get("synthesis", {})
    preset_names = list(synthesis_cfg.get("preset_names", []))
    preset_name = rng.choice(preset_names) if preset_names else "default"
    presets = synthesis_cfg.get("presets", {})
    preset_cfg = presets.get(preset_name, {}) if isinstance(presets, dict) else {}

    beta_values = preset_cfg.get("beta_range", synthesis_cfg.get("beta_range", [0.35, 1.2]))
    beta = rng.uniform(float(beta_values[0]), float(beta_values[1]))

    glare_cfg = synthesis_cfg.get("glare", {})
    glare_enabled = bool(glare_cfg.get("enabled", True))
    glare_probability = float(preset_cfg.get("glare_probability", glare_cfg.get("probability", 0.5)))
    glare_strength_range = glare_cfg.get("strength_range", [0.03, 0.12])
    glare_strength = rng.uniform(float(glare_strength_range[0]), float(glare_strength_range[1]))
    use_glare = glare_enabled and rng.random() < glare_probability

    noise_cfg = synthesis_cfg.get("noise", {})
    noise_enabled = bool(noise_cfg.get("enabled", True))
    noise_mode = str(noise_cfg.get("mode", "gaussian"))
    noise_strength_range = preset_cfg.get("noise_strength_range", noise_cfg.get("strength_range", [0.0, 0.03]))
    noise_strength = rng.uniform(float(noise_strength_range[0]), float(noise_strength_range[1]))
    use_noise = noise_enabled and noise_strength > 0

    return {
        "preset": preset_name,
        "beta": beta,
        "airlight": sample_airlight(rng, synthesis_cfg.get("airlight_range", {})),
        "glare": {
            "enabled": use_glare,
            "strength": glare_strength if use_glare else 0.0,
            "spot_count": rng.randint(
                int(glare_cfg.get("spot_count_range", [1, 3])[0]),
                int(glare_cfg.get("spot_count_range", [1, 3])[1]),
            ),
            "sigma_ratio": rng.uniform(
                float(glare_cfg.get("sigma_ratio_range", [0.08, 0.18])[0]),
                float(glare_cfg.get("sigma_ratio_range", [0.08, 0.18])[1]),
            ),
        },
        "noise": {
            "enabled": use_noise,
            "mode": noise_mode,
            "strength": noise_strength if use_noise else 0.0,
        },
    }


def synthesize_sample(
    source_path: Path,
    split: str,
    sample_index: int,
    config: dict,
    depth_estimator,
    rng: random.Random,
) -> SynthesisSample:
    synthesis_cfg = config.get("synthesis", {})
    image_size = tuple(int(value) for value in synthesis_cfg.get("image_size", [256, 256]))
    target_image = Image.open(source_path).convert("RGB").resize(image_size, Image.BICUBIC)
    target_tensor = pil_to_tensor(target_image).unsqueeze(0)

    depth = depth_estimator(target_tensor, source_path)
    params = sample_degradation_params(config, rng)
    transmission = transmission_from_depth(depth, params["beta"]).clamp(
        min=float(synthesis_cfg.get("t_min", 0.15)),
        max=1.0,
    )

    airlight_values = [float(value) for value in params["airlight"]]
    airlight_tensor = airlight_to_tensor(airlight_values, device=target_tensor.device, dtype=target_tensor.dtype)
    airlight_map = airlight_tensor_to_broadcast(airlight_tensor, spatial_size=target_tensor.shape[2:])
    degraded = target_tensor * transmission + airlight_map * (1.0 - transmission)

    glare_cfg = params["glare"]
    if glare_cfg["enabled"]:
        degraded = apply_glare(
            degraded,
            strength=float(glare_cfg["strength"]),
            rng=rng,
            spot_count=int(glare_cfg["spot_count"]),
            sigma_ratio=float(glare_cfg["sigma_ratio"]),
        )

    noise_cfg = params["noise"]
    if noise_cfg["enabled"]:
        degraded = apply_noise(degraded, mode=str(noise_cfg["mode"]), strength=float(noise_cfg["strength"]))

    sample_id = f"{source_path.stem}-s{sample_index + 1:02d}"
    metadata = {
        "sample_id": sample_id,
        "split": split,
        "source_image": source_path.name,
        "source_path": source_path.as_posix(),
        "beta": round(float(params["beta"]), 6),
        "A": [round(value, 6) for value in airlight_values],
        "depth_source": str(config.get("model", {}).get("depth_backend", "placeholder_inverse_luminance")),
        "depth_origin_path": None,
        "preset": params["preset"],
        "noise": {
            "enabled": bool(noise_cfg["enabled"]),
            "mode": str(noise_cfg["mode"]),
            "strength": round(float(noise_cfg["strength"]), 6),
        },
        "glare": {
            "enabled": bool(glare_cfg["enabled"]),
            "strength": round(float(glare_cfg["strength"]), 6),
            "spot_count": int(glare_cfg["spot_count"]),
            "sigma_ratio": round(float(glare_cfg["sigma_ratio"]), 6),
        },
        "transmission_min": round(float(transmission.min().item()), 6),
        "transmission_max": round(float(transmission.max().item()), 6),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "seed": int(config.get("runtime", {}).get("seed", 0)),
    }
    return SynthesisSample(
        sample_id=sample_id,
        split=split,
        target=target_tensor,
        depth=depth,
        depth_origin_path=None,
        transmission=transmission,
        airlight=airlight_values,
        input_image=degraded.clamp(0.0, 1.0),
        metadata=metadata,
    )


def save_synthesis_sample(
    sample: SynthesisSample,
    synthetic_root: Path,
    output_format: str,
    save_depth: bool,
) -> dict[str, Path]:
    split_root = synthetic_root / sample.split
    input_dir = split_root / "input"
    target_dir = split_root / "target"
    depth_dir = split_root / "depth"
    transmission_dir = split_root / "transmission"
    airlight_dir = split_root / "airlight"
    metadata_dir = split_root / "metadata"
    directories = [input_dir, target_dir, transmission_dir, airlight_dir, metadata_dir]
    if save_depth:
        directories.append(depth_dir)
    for directory in directories:
        ensure_dir(directory)

    image_suffix = output_format if output_format.startswith(".") else f".{output_format}"
    input_path = input_dir / f"{sample.sample_id}{image_suffix}"
    target_path = target_dir / f"{sample.sample_id}{image_suffix}"
    depth_path = depth_dir / f"{sample.sample_id}{image_suffix}"
    transmission_path = transmission_dir / f"{sample.sample_id}{image_suffix}"
    airlight_path = airlight_dir / f"{sample.sample_id}.json"
    metadata_path = metadata_dir / f"{sample.sample_id}.json"

    save_image(_tensor_to_image(sample.input_image, mode="RGB"), input_path)
    save_image(_tensor_to_image(sample.target, mode="RGB"), target_path)
    if save_depth:
        save_image(_tensor_to_image(sample.depth, mode="L"), depth_path)
    save_image(_tensor_to_image(sample.transmission, mode="L"), transmission_path)
    write_json(airlight_path, {"A": [round(value, 6) for value in sample.airlight]})
    write_json(metadata_path, sample.metadata)

    return {
        "input": input_path,
        "target": target_path,
        "depth": depth_path,
        "transmission": transmission_path,
        "airlight": airlight_path,
        "metadata": metadata_path,
    }


def run_synthesis_pipeline(config: dict, project_root: Path) -> SynthesisSummary:
    from src.utils.io import resolve_project_path

    data_cfg = config.get("data", {})
    runtime_cfg = config.get("runtime", {})
    output_cfg = config.get("output", {})
    synthesis_cfg = config.get("synthesis", {})

    clean_root = resolve_project_path(data_cfg["clean_source_root"], project_root)
    raw_root = resolve_project_path(data_cfg["raw_field_root"], project_root)
    synthetic_root = resolve_project_path(data_cfg["synthetic_root"], project_root)
    split_root = resolve_project_path(data_cfg["split_dir"], project_root)
    manifest_path = resolve_project_path(output_cfg["manifest_path"], project_root)

    clean_images = list_image_files(
        clean_root / "images",
        recursive=bool(data_cfg.get("recursive", False)),
        allowed_extensions=data_cfg.get("image_extensions"),
    )
    raw_images = list_image_files(
        raw_root / "images",
        recursive=bool(data_cfg.get("recursive", False)),
        allowed_extensions=data_cfg.get("image_extensions"),
    )
    split_plan = build_split_plan(
        clean_images,
        split_ratio=synthesis_cfg.get("split_ratio", {"train": 0.7, "val": 0.2, "test": 0.1}),
        seed=int(runtime_cfg.get("seed", 0)),
    )
    samples_per_image = int(synthesis_cfg.get("samples_per_image", 1))
    dry_run = bool(runtime_cfg.get("dry_run", False))
    output_image_format = str(data_cfg.get("output_image_format", ".png"))

    if dry_run:
        return SynthesisSummary(
            clean_images=len(clean_images),
            raw_images=len(raw_images),
            synthetic_train=len(split_plan["train"]) * samples_per_image,
            synthetic_val=len(split_plan["val"]) * samples_per_image,
            synthetic_test=len(split_plan["test"]) * samples_per_image,
            generated_total=len(clean_images) * samples_per_image,
            manifest_path=str(manifest_path),
        )

    ensure_dir(split_root)
    if bool(output_cfg.get("overwrite", False)):
        for split in ("train", "val", "test"):
            _clear_directory(synthetic_root / split)
            split_file = split_root / f"{split}.txt"
            if split_file.exists():
                split_file.unlink()

    rng = random.Random(int(runtime_cfg.get("seed", 0)))
    depth_estimator = build_depth_estimator(config, project_root=project_root)
    generated_counts = {"train": 0, "val": 0, "test": 0}
    preset_counts: dict[str, int] = {}
    split_entries = {"train": [], "val": [], "test": []}
    manifest_samples: list[dict[str, Any]] = []
    clean_manifest = _write_source_manifest(clean_root, clean_images, project_root, "clean_source_index.json")
    raw_manifest = _write_source_manifest(raw_root, raw_images, project_root, "raw_field_index.json")
    save_depth = bool(data_cfg.get("save_depth", True))

    for split, paths in split_plan.items():
        for source_path in paths:
            for sample_index in range(samples_per_image):
                sample = synthesize_sample(source_path, split, sample_index, config, depth_estimator, rng)
                output_paths = save_synthesis_sample(sample, synthetic_root, output_image_format, save_depth=save_depth)
                preset_name = str(sample.metadata["preset"])
                preset_counts[preset_name] = preset_counts.get(preset_name, 0) + 1
                generated_counts[split] += 1
                split_entries[split].append(f"synthetic/{split}/{sample.sample_id}")

                sample.metadata.update(
                    {
                        "source_image": _relative_posix(source_path, project_root),
                        "target_image": _relative_posix(output_paths["target"], project_root),
                        "depth_image": _relative_posix(output_paths["depth"], project_root) if save_depth else None,
                        "input_image": _relative_posix(output_paths["input"], project_root),
                        "transmission_image": _relative_posix(output_paths["transmission"], project_root),
                        "airlight_file": _relative_posix(output_paths["airlight"], project_root),
                        "metadata_file": _relative_posix(output_paths["metadata"], project_root),
                    }
                )
                write_json(output_paths["metadata"], sample.metadata)
                manifest_samples.append(sample.metadata)

        split_file = split_root / f"{split}.txt"
        split_file.write_text("\n".join(split_entries[split]), encoding="utf-8")

    manifest_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": config.get("project", {}),
        "config_snapshot": config,
        "clean_images": len(clean_images),
        "raw_images": len(raw_images),
        "samples_per_image": samples_per_image,
        "generated_counts": generated_counts,
        "split_source_counts": {key: len(value) for key, value in split_plan.items()},
        "preset_counts": preset_counts,
        "synthetic_root": _relative_posix(synthetic_root, project_root),
        "split_dir": _relative_posix(split_root, project_root),
        "source_manifests": {
            "clean_source": _relative_posix(clean_manifest, project_root),
            "raw_field": _relative_posix(raw_manifest, project_root),
        },
        "samples": manifest_samples,
    }
    write_json(manifest_path, manifest_payload)
    return SynthesisSummary(
        clean_images=len(clean_images),
        raw_images=len(raw_images),
        synthetic_train=generated_counts["train"],
        synthetic_val=generated_counts["val"],
        synthetic_test=generated_counts["test"],
        generated_total=sum(generated_counts.values()),
        manifest_path=str(manifest_path),
    )
