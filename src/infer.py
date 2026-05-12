"""Inference entrypoint for the physical-guided framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import InferImageDataset
from src.inference.pipeline import build_model, resolve_device, run_single_image
from src.utils.config import load_config
from src.utils.io import ensure_dir, resolve_project_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physical-guided image enhancement inference.")
    parser.add_argument(
        "--config",
        default="configs/infer/infer_base.yaml",
        help="Path to the inference config file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_project_path(args.config, project_root)
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    output_cfg = config["output"]
    runtime_cfg = config.get("runtime", {})

    input_path = resolve_project_path(data_cfg.get("input_path", "./input_images"), project_root)
    output_dir = resolve_project_path(output_cfg.get("output_dir", "./outputs/infer"), project_root)
    recursive = bool(data_cfg.get("recursive", False))
    save_compare = bool(output_cfg.get("save_compare", True))
    compare_suffix = str(output_cfg.get("compare_suffix", "_compare"))
    enhanced_suffix = str(output_cfg.get("enhanced_suffix", "_enhanced"))
    max_images = int(runtime_cfg.get("max_images", 0))

    input_size_cfg = data_cfg.get("image_size", [512, 512])
    input_size = (int(input_size_cfg[0]), int(input_size_cfg[1]))

    ensure_dir(output_dir)
    dataset = InferImageDataset(input_path, recursive=recursive)
    image_files = [dataset[index] for index in range(len(dataset))]
    if max_images > 0:
        image_files = image_files[:max_images]

    device = resolve_device(model_cfg)
    model = build_model(model_cfg, project_root).to(device)
    model.eval()

    print(f"[INFO] Config      : {config_path}")
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Input       : {input_path}")
    print(f"[INFO] Output      : {output_dir}")
    print(f"[INFO] Input size  : {input_size}")
    print(f"[INFO] Image count : {len(image_files)}")

    for image_path in image_files:
        result = run_single_image(
            image_path=image_path,
            model=model,
            input_size=input_size,
            output_dir=output_dir,
            save_compare=save_compare,
            enhanced_suffix=enhanced_suffix,
            compare_suffix=compare_suffix,
            device=device,
        )
        print(f"[OK] Enhanced: {result['enhanced']}")
        if "compare" in result:
            print(f"[OK] Compare : {result['compare']}")

    print(f"[INFO] Inference finished for {len(image_files)} image(s).")


if __name__ == "__main__":
    main()
