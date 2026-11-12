"""
mrRemBG Command Line Runner.

Can be executed standalone, inside WSL2 Ubuntu, or invoked directly by Meshroom.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure rembg_core can be imported
cli_dir = Path(__file__).resolve().parent
plugin_root = cli_dir.parent
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from rembg_core.engine import RemBGEngine
from rembg_core.sfm_io import SfmDataIO, SfmViewItem, normalize_path_for_current_os, read_image_dimensions


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("mr_rembg_runner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[mrRemBG CLI] %(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Meshroom RemBG Background Removal Runner")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths or sfmData .sfm path")
    parser.add_argument("--output-rgba-dir", type=Path, required=True, help="Directory to save transparent RGBA images")
    parser.add_argument("--output-mask-dir", type=Path, required=True, help="Directory to save binary/grayscale masks")
    parser.add_argument("--output-sfm", type=Path, default=None, help="Optional path to output updated sfmData JSON")
    parser.add_argument("--original-sfm", type=Path, default=None, help="Optional path to original sfmData JSON")
    parser.add_argument("--model", type=str, default="u2net", help="rembg model name (u2net, u2netp, etc.)")
    parser.add_argument("--alpha-matting", action="store_true", help="Enable alpha matting")
    parser.add_argument("--af-thresh", type=int, default=240, help="Alpha matting foreground threshold")
    parser.add_argument("--ab-thresh", type=int, default=10, help="Alpha matting background threshold")
    parser.add_argument("--erode-size", type=int, default=10, help="Alpha matting erode size")
    parser.add_argument("--pad-to-square", action="store_true", help="Pad resulting image to square 1:1")

    args = parser.parse_args()
    logger = setup_logger()

    logger.info("Initializing mrRemBG runner with model '%s'...", args.model)
    args.output_rgba_dir.mkdir(parents=True, exist_ok=True)
    args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    io_handler = SfmDataIO(logger=logger)
    engine = RemBGEngine(
        model_name=args.model,
        alpha_matting=args.alpha_matting,
        af_thresh=args.af_thresh,
        ab_thresh=args.ab_thresh,
        erode_size=args.erode_size,
        logger=logger,
    )

    # Resolve views to process
    views: list[SfmViewItem] = []
    original_dict = None

    if len(args.images) == 1 and Path(args.images[0]).is_file() and Path(args.images[0]).suffix.lower() in (".sfm", ".json"):
        views, original_dict = io_handler.load_views(args.images[0])
    elif args.original_sfm and args.original_sfm.is_file():
        views, original_dict = io_handler.load_views(str(args.original_sfm))
    else:
        for idx, img_str in enumerate(args.images):
            img_p = normalize_path_for_current_os(img_str).resolve()
            if img_p.is_file():
                w, h = read_image_dimensions(img_p)
                views.append(SfmViewItem(
                    view_id=str(idx),
                    pose_id=str(idx),
                    intrinsic_id="0",
                    image_path=img_p,
                    width=w,
                    height=h,
                    raw_dict={"viewId": str(idx), "poseId": str(idx), "intrinsicId": "0", "path": str(img_p), "width": str(w), "height": str(h)}
                ))

    if not views:
        logger.error("No valid views or images found to process!")
        return 1

    logger.info("Processing %d views...", len(views))
    processed_records = []

    for idx, item in enumerate(views):
        base_name = item.image_path.stem
        # Generate predictable output names
        rgba_out = args.output_rgba_dir / f"{base_name}_rgba.png"
        mask_out = args.output_mask_dir / f"{base_name}_mask.png"

        logger.info("[%d/%d] Processing %s -> %s", idx + 1, len(views), item.image_path.name, rgba_out.name)
        engine.process_image(
            input_image_path=item.image_path,
            output_rgba_path=rgba_out,
            output_mask_path=mask_out,
            pad_to_square=args.pad_to_square,
        )
        processed_records.append((item, rgba_out, mask_out))

    if args.output_sfm:
        io_handler.write_output_sfm(
            output_sfm_path=args.output_sfm,
            original_sfm_dict=original_dict,
            processed_views=processed_records,
        )

    logger.info("mrRemBG execution completed successfully for all %d images.", len(views))
    return 0


if __name__ == "__main__":
    sys.exit(main())

