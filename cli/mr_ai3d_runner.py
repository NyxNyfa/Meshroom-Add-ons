"""
mrAIGen3D Command Line Runner.

Executes TripoSR, TripoSplat, or Remote Colab inference.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

cli_dir = Path(__file__).resolve().parent
plugin_root = cli_dir.parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from mrai3d_core import get_triposr_runner, get_triposplat_runner, ColabBridgeClient
from mrai3d_core.sfm_helper import normalize_path_for_current_os


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mr_ai3d_runner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[mrAIGen3D CLI] %(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="mrAIGen3D Single-Image 3D Reconstruction Runner")
    parser.add_argument("--image", type=Path, required=True, help="Input single image file")
    parser.add_argument("--output-obj", type=Path, required=True, help="Path for output OBJ mesh")
    parser.add_argument("--output-glb", type=Path, required=True, help="Path for output GLB mesh")
    parser.add_argument("--output-ply", type=Path, required=True, help="Path for output Gaussian PLY")
    parser.add_argument("--model", type=str, default="triposr", help="Model: triposr, triposplat, spar3d, etc.")
    parser.add_argument("--mc-resolution", type=int, default=192, help="Marching cubes resolution (192, 256)")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Evaluation chunk size (4096, 8192)")
    parser.add_argument("--num-gaussians", type=int, default=131072, help="Gaussian count for TripoSplat")
    parser.add_argument("--remote-url", type=str, default="", help="Remote Colab endpoint URL if applicable")
    parser.add_argument("--hf-token", type=str, default="", help="Optional Hugging Face token")

    args = parser.parse_args()
    logger = setup_logger()

    args.image = normalize_path_for_current_os(args.image).resolve()
    logger.info("Starting mrAIGen3D with model '%s' on %s...", args.model, args.image)
    if not args.image.is_file():
        logger.error("Image file does not exist: %s", args.image)
        return 1

    model_lower = args.model.lower()

    if model_lower == "triposplat":
        TripoSplatRunner = get_triposplat_runner()
        runner = TripoSplatRunner(
            num_gaussians=args.num_gaussians,
            logger=logger,
        )
        runner.run(
            image_path=args.image,
            output_ply=args.output_ply,
        )
        logger.info("TripoSplat generation completed.")
        return 0

    elif model_lower == "triposr":
        TripoSRRunner = get_triposr_runner()
        runner = TripoSRRunner(
            chunk_size=args.chunk_size,
            mc_resolution=args.mc_resolution,
            logger=logger,
        )
        runner.run(
            image_path=args.image,
            output_obj_path=args.output_obj,
            output_glb_path=args.output_glb,
        )
        logger.info("TripoSR generation completed.")
        return 0

    else:
        # Remote Colab models (SF3D, SPAR3D, Trellis, Hunyuan3D-2.1)
        if not args.remote_url:
            logger.error("Model '%s' requires a remote Colab endpoint URL (--remote-url).", args.model)
            return 1

        client = ColabBridgeClient(endpoint_url=args.remote_url, logger=logger)
        client.generate(
            image_path=args.image,
            output_dir=args.output_obj.parent,
            model_name=args.model,
            hf_token=args.hf_token,
        )
        logger.info("Remote Colab generation completed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
