"""Stage-one inference entrypoint.

Usage:
    conda run -n DL2 python -m src.infer
    conda run -n DL2 python -m src.infer --config ./configs/infer_base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.pipeline import build_model, resolve_device, run_single_image
from src.utils.config import load_config
from src.utils.io import ensure_dir, list_image_files, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-one image enhancement inference.")
    parser.add_argument(
        "--config",
        default="configs/infer_base.yaml",
        help="Path to the inference config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = resolve_project_path(args.config, project_root)
    config = load_config(config_path)

    input_cfg = config.get("input", {})
    model_cfg = config.get("model", {})
    output_cfg = config.get("output", {})

    input_path = resolve_project_path(input_cfg.get("path", "./input_images"), project_root)
    output_dir = resolve_project_path(output_cfg.get("dir", "./outputs/infer"), project_root)
    recursive = bool(input_cfg.get("recursive", False))
    allowed_extensions = input_cfg.get("allowed_extensions")
    save_compare = bool(output_cfg.get("save_compare", True))
    compare_suffix = str(output_cfg.get("compare_suffix", "_compare"))
    enhanced_suffix = str(output_cfg.get("enhanced_suffix", "_enhanced"))

    input_size_cfg = model_cfg.get("input_size", [512, 512])
    if not isinstance(input_size_cfg, list) or len(input_size_cfg) != 2:
        raise ValueError("model.input_size must be a two-element list, for example [512, 512].")
    input_size = (int(input_size_cfg[0]), int(input_size_cfg[1]))

    ensure_dir(output_dir)
    image_files = list_image_files(input_path, recursive=recursive, allowed_extensions=allowed_extensions)
    if not image_files:
        raise FileNotFoundError(f"No supported images found under: {input_path}")

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


if __name__ == "__main__":
    main()
