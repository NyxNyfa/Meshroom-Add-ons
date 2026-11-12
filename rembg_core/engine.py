"""
RemBG Matting Engine for mrRemBG.

Uses `rembg` (u2net, u2netp, isnet, etc.) with session caching and optional
alpha matting, padding to square, and mask separation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


class RemBGEngine:
    """Core image matting engine."""

    def __init__(
        self,
        model_name: str = "u2net",
        alpha_matting: bool = False,
        af_thresh: int = 240,
        ab_thresh: int = 10,
        erode_size: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.alpha_matting = alpha_matting
        self.af_thresh = af_thresh
        self.ab_thresh = ab_thresh
        self.erode_size = erode_size
        self.logger = logger or logging.getLogger("mrRemBG.engine")
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                import rembg
                self.logger.info("Initializing rembg session with model: %s", self.model_name)
                self._session = rembg.new_session(self.model_name)
            except Exception as e:
                self.logger.error("Failed to initialize rembg session: %s", e)
                raise
        return self._session

    def process_image(
        self,
        input_image_path: Path,
        output_rgba_path: Path,
        output_mask_path: Path,
        pad_to_square: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Process a single image, generating RGBA transparent image and binary/alpha mask.
        """
        self.logger.info("Processing background removal on: %s", input_image_path)
        with Image.open(input_image_path) as raw_img:
            img = ImageOps.exif_transpose(raw_img)
            orig_mode = img.mode

        import rembg
        session = self._get_session()

        rgba_res = rembg.remove(
            img,
            session=session,
            alpha_matting=self.alpha_matting,
            alpha_matting_foreground_threshold=self.af_thresh,
            alpha_matting_background_threshold=self.ab_thresh,
            alpha_matting_erode_size=self.erode_size,
        )

        if not isinstance(rgba_res, Image.Image):
            rgba_res = Image.fromarray(rgba_res)
        rgba_res = rgba_res.convert("RGBA")

        # Extract alpha mask
        alpha = rgba_res.split()[3]

        if pad_to_square:
            rgba_res = self._pad_image_to_square(rgba_res)
            alpha = self._pad_image_to_square(alpha, fill=0)

        output_rgba_path.parent.mkdir(parents=True, exist_ok=True)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        rgba_res.save(output_rgba_path, format="PNG")
        alpha.save(output_mask_path, format="PNG")

        self.logger.info("Saved RGBA: %s | Mask: %s", output_rgba_path, output_mask_path)
        return output_rgba_path, output_mask_path

    @staticmethod
    def _pad_image_to_square(img: Image.Image, fill: Any = (0, 0, 0, 0)) -> Image.Image:
        """Pad image to square maintaining aspect ratio and centering the content."""
        w, h = img.size
        if w == h:
            return img
        max_dim = max(w, h)
        if img.mode == "RGBA":
            new_img = Image.new("RGBA", (max_dim, max_dim), fill if isinstance(fill, tuple) else (0, 0, 0, 0))
        elif img.mode == "L":
            new_img = Image.new("L", (max_dim, max_dim), fill if isinstance(fill, int) else 0)
        else:
            new_img = Image.new(img.mode, (max_dim, max_dim), fill)

        offset_x = (max_dim - w) // 2
        offset_y = (max_dim - h) // 2
        new_img.paste(img, (offset_x, offset_y))
        return new_img

