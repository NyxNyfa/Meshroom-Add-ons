"""
SfMData Parser and View Selector for mrAIGen3D.

Reads AliceVision .sfm JSON from CameraInit or mrRemBG and selects the target single view.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def normalize_path_for_current_os(p: Path | str) -> Path:
    """
    Ensure path is valid for the current OS environment:
    - If running inside POSIX (Linux/WSL): converts Windows 'D:/...' to '/mnt/d/...'.
    - If running inside Windows: converts WSL '/mnt/d/...' to 'D:/...'.
    """
    p_str = str(p).strip()
    if not p_str:
        return Path("")
    if os.name == "posix":
        match = re.match(r"^([a-zA-Z]):[\\/](.*)", p_str)
        if match:
            drive = match.group(1).lower()
            rest = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
        return Path(p_str)
    else:
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)", p_str)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2)
            return Path(f"{drive}:/{rest}")
        return Path(p_str)


def resolve_single_image(
    input_sfm_str: str,
    override_image_str: str = "",
    selected_view_index: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Optional[str]]:
    """
    Resolve exactly one target image path from override or SfMData.
    Returns:
        Tuple of (resolved Path, optional view_id string)
    """
    log = logger or logging.getLogger("mrAIGen3D.sfm_helper")

    if override_image_str and override_image_str.strip():
        direct_path = normalize_path_for_current_os(override_image_str.strip()).resolve()
        if not direct_path.is_file():
            raise FileNotFoundError(f"Override image path does not exist: {direct_path}")
        log.info("Resolved image from override path: %s", direct_path)
        return direct_path, "0"

    sfm_path = normalize_path_for_current_os(input_sfm_str.strip()).resolve()
    if not sfm_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {sfm_path}")

    # Direct image file passed as input
    if sfm_path.is_file() and sfm_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        log.info("Resolved single image directly: %s", sfm_path)
        return sfm_path, "0"

    # Directory of images passed as input
    if sfm_path.is_dir():
        imgs = sorted(
            [p for p in sfm_path.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")],
            key=lambda p: p.name.lower()
        )
        if not imgs:
            raise ValueError(f"No images found in directory: {sfm_path}")
        idx = max(0, min(selected_view_index, len(imgs) - 1))
        chosen = imgs[idx]
        log.info("Selected image [%d/%d] from directory: %s", idx + 1, len(imgs), chosen)
        return chosen, str(idx)

    # SfMData JSON file (.sfm)
    if sfm_path.is_file() and sfm_path.suffix.lower() in (".sfm", ".json"):
        try:
            data = json.loads(sfm_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse SfMData file {sfm_path}: {e}") from e

        views = data.get("views", [])
        if not views:
            raise ValueError(f"No views found in SfMData file: {sfm_path}")

        idx = max(0, min(selected_view_index, len(views) - 1))
        view_entry = views[idx]
        raw_img_path = view_entry.get("path", "")
        if not raw_img_path:
            raise ValueError(f"View at index {idx} in {sfm_path} has no 'path' attribute.")

        img_path = normalize_path_for_current_os(raw_img_path).resolve()
        if not img_path.is_file():
            # Check relative to sfm file
            alt_path = (sfm_path.parent / raw_img_path).resolve()
            if alt_path.is_file():
                img_path = alt_path
            else:
                # Check candidate locations by filename
                fname = Path(raw_img_path).name
                candidates = [
                    Path.cwd() / fname,
                    sfm_path.parent / fname,
                    Path("/mnt/d/Meshroom/griffin" if os.name == "posix" else "D:/Meshroom/griffin") / fname,
                    Path("/mnt/d/Meshroom" if os.name == "posix" else "D:/Meshroom") / fname,
                    Path("/mnt/d/Download" if os.name == "posix" else "D:/Download") / fname,
                ]
                found = False
                for cand in candidates:
                    if cand.is_file():
                        log.info("Auto-relocated image '%s' to: %s", fname, cand)
                        img_path = cand
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(f"Image for view {idx} not found at {img_path}")

        view_id = str(view_entry.get("viewId", idx))
        log.info("Selected view [%d/%d] (viewId: %s) from SfMData: %s", idx + 1, len(views), view_id, img_path)
        return img_path, view_id

    raise ValueError(f"Unsupported input format for mrAIGen3D: {sfm_path}")

