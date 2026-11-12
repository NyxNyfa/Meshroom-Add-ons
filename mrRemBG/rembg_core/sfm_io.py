"""
SfMData Parser and Output Generator for mrRemBG.

Handles parsing input views from:
- AliceVision SfMData JSON (.sfm)
- Directory of images (.jpg, .png, etc.)
- Single image file
And writes updated SfMData JSON (.sfm) pointing to the segmented RGBA images.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def read_image_dimensions(image_path: Path | str) -> Tuple[int, int]:
    """
    Read width and height of an image without requiring third-party libraries (e.g. PIL).
    Parses headers for PNG, JPEG, WebP, and BMP. Falls back to PIL if available, or (0, 0).
    """
    p = Path(image_path)
    if not p.is_file():
        return (0, 0)

    try:
        with open(p, "rb") as f:
            header = f.read(32)
            # PNG: bytes 16-24 hold width and height as big-endian 32-bit unsigned ints
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                import struct
                w, h = struct.unpack(">II", header[16:24])
                return int(w), int(h)

            # BMP: bytes 18-26 hold width and height as little-endian 32-bit signed ints
            if header.startswith(b"BM") and len(header) >= 26:
                import struct
                w, h = struct.unpack("<ii", header[18:26])
                return int(abs(w)), int(abs(h))

            # JPEG: scan markers until SOF0/SOF1/SOF2
            if header.startswith(b"\xFF\xD8"):
                import struct
                f.seek(2)
                while True:
                    marker_bytes = f.read(2)
                    if len(marker_bytes) < 2:
                        break
                    marker, = struct.unpack(">H", marker_bytes)
                    if marker in (0xFFC0, 0xFFC1, 0xFFC2):
                        f.read(3)  # length (2) + precision (1)
                        h, w = struct.unpack(">HH", f.read(4))
                        return int(w), int(h)
                    elif 0xFFD0 <= marker <= 0xFFD9 or marker == 0xFF01:
                        continue
                    else:
                        len_bytes = f.read(2)
                        if len(len_bytes) < 2:
                            break
                        seg_len, = struct.unpack(">H", len_bytes)
                        f.seek(seg_len - 2, 1)

            # WebP
            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                import struct
                tag = header[12:16]
                if tag == b"VP8 ":
                    f.seek(26)
                    w, h = struct.unpack("<HH", f.read(4))
                    return int(w & 0x3FFF), int(h & 0x3FFF)
                elif tag == b"VP8L":
                    f.seek(21)
                    b0, b1, b2, b3 = f.read(4)
                    w = 1 + (((b1 & 0x3F) << 8) | b0)
                    h = 1 + (((b3 & 0xF) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
                    return int(w), int(h)
                elif tag == b"VP8X":
                    f.seek(24)
                    b = f.read(6)
                    w = 1 + (b[0] | (b[1] << 8) | (b[2] << 16))
                    h = 1 + (b[3] | (b[4] << 8) | (b[5] << 16))
                    return int(w), int(h)
    except Exception:
        pass

    try:
        from PIL import Image
        with Image.open(p) as img:
            return int(img.width), int(img.height)
    except Exception:
        pass

    return (0, 0)


def normalize_to_windows_path(p: Path | str) -> str:
    """Normalize any path (including WSL /mnt/d/...) to a valid Windows path (D:/...)."""
    p_str = str(p)
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)", p_str)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2)
        return f"{drive}:/{rest}"
    return p_str.replace("\\", "/")


def normalize_to_wsl_path(p: Path | str) -> str:
    """Normalize any path (including Windows D:/...) to a valid WSL path (/mnt/d/...)."""
    p_str = str(p)
    match = re.match(r"^([a-zA-Z]):[\\/](.*)", p_str)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p_str.replace("\\", "/")


def normalize_path_for_current_os(p: Path | str) -> Path:
    """
    Ensure path is valid for the current OS environment:
    - If running inside POSIX (Linux/WSL): converts Windows 'D:/...' to '/mnt/d/...'.
    - If running inside Windows: converts WSL '/mnt/d/...' to 'D:/...'.
    """
    p_str = str(p).strip()
    if not p_str:
        return Path("")
    import os
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


@dataclass
class SfmViewItem:
    """Represents a single view item for processing."""
    view_id: str
    pose_id: str
    intrinsic_id: str
    image_path: Path
    width: int
    height: int
    raw_dict: Optional[Dict[str, Any]] = None


class SfmDataIO:
    """I/O handler for AliceVision SfMData and image collections."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("mrRemBG.sfm_io")

    def load_views(self, input_path_str: str, override_image_str: str = "") -> Tuple[List[SfmViewItem], Optional[Dict[str, Any]]]:
        """
        Parse views from SfMData .sfm, directory, or direct image file.
        Returns:
            Tuple of (list of SfmViewItem, raw SfMData dict if available)
        """
        if override_image_str and override_image_str.strip():
            single_path = normalize_path_for_current_os(override_image_str.strip()).resolve()
            if not single_path.is_file():
                raise FileNotFoundError(f"Override image path does not exist: {single_path}")
            self.logger.info("Using override single image: %s", single_path)
            w, h = read_image_dimensions(single_path)
            item = SfmViewItem(
                view_id="0",
                pose_id="0",
                intrinsic_id="0",
                image_path=single_path,
                width=w,
                height=h,
                raw_dict={"viewId": "0", "poseId": "0", "intrinsicId": "0", "path": str(single_path), "width": str(w), "height": str(h)}
            )
            return [item], None

        input_path = normalize_path_for_current_os(input_path_str.strip()).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        # Case 1: SfMData JSON file (.sfm)
        if input_path.is_file() and input_path.suffix.lower() in (".sfm", ".json"):
            return self._load_from_sfm_json(input_path)

        # Case 2: Single image file
        if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            w, h = read_image_dimensions(input_path)
            item = SfmViewItem(
                view_id="0",
                pose_id="0",
                intrinsic_id="0",
                image_path=input_path,
                width=w,
                height=h,
                raw_dict={"viewId": "0", "poseId": "0", "intrinsicId": "0", "path": str(input_path), "width": str(w), "height": str(h)}
            )
            return [item], None

        # Case 3: Directory of images
        if input_path.is_dir():
            return self._load_from_directory(input_path)

        raise ValueError(f"Unsupported input format for mrRemBG: {input_path}")

    def _load_from_sfm_json(self, sfm_file: Path) -> Tuple[List[SfmViewItem], Dict[str, Any]]:
        self.logger.info("Loading SfMData JSON from: %s", sfm_file)
        try:
            data = json.loads(sfm_file.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse SfMData JSON file {sfm_file}: {e}") from e

        if not isinstance(data, dict) or "views" not in data:
            raise ValueError(f"SfMData missing 'views' array in {sfm_file}")

        views: List[SfmViewItem] = []
        for idx, v in enumerate(data.get("views", [])):
            if not isinstance(v, dict):
                continue
            raw_path = v.get("path", "")
            if not raw_path:
                continue
            img_path = normalize_path_for_current_os(raw_path).resolve()
            if not img_path.is_file():
                # 1. Check relative to sfm file
                alt_path = (sfm_file.parent / raw_path).resolve()
                if alt_path.is_file():
                    img_path = alt_path
                else:
                    # 2. Check relocated candidates by filename
                    fname = Path(raw_path).name
                    import os
                    candidates = [
                        Path.cwd() / fname,
                        sfm_file.parent / fname,
                        Path("/mnt/d/Meshroom/griffin" if os.name == "posix" else "D:/Meshroom/griffin") / fname,
                        Path("/mnt/d/Meshroom" if os.name == "posix" else "D:/Meshroom") / fname,
                        Path("/mnt/d/Download" if os.name == "posix" else "D:/Download") / fname,
                    ]
                    for cand in candidates:
                        if cand.is_file():
                            self.logger.info("Auto-relocated image '%s' to: %s", fname, cand)
                            img_path = cand
                            break

            view_id = str(v.get("viewId", idx))
            pose_id = str(v.get("poseId", view_id))
            intrinsic_id = str(v.get("intrinsicId", "0"))
            w = int(v.get("width", 0))
            h = int(v.get("height", 0))

            if w == 0 or h == 0:
                if img_path.is_file():
                    w, h = read_image_dimensions(img_path)

            views.append(SfmViewItem(
                view_id=view_id,
                pose_id=pose_id,
                intrinsic_id=intrinsic_id,
                image_path=img_path,
                width=w,
                height=h,
                raw_dict=dict(v)
            ))

        self.logger.info("Loaded %d views from SfMData", len(views))
        return views, data

    def _load_from_directory(self, dir_path: Path) -> Tuple[List[SfmViewItem], Optional[Dict[str, Any]]]:
        self.logger.info("Scanning image directory: %s", dir_path)
        all_files = sorted(
            [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
            key=lambda p: p.name.lower()
        )
        if not all_files:
            raise ValueError(f"No supported images ({', '.join(SUPPORTED_EXTENSIONS)}) found in {dir_path}")

        views: List[SfmViewItem] = []
        for idx, img_path in enumerate(all_files):
            w, h = read_image_dimensions(img_path)

            views.append(SfmViewItem(
                view_id=str(idx),
                pose_id=str(idx),
                intrinsic_id="0",
                image_path=img_path,
                width=w,
                height=h,
                raw_dict={"viewId": str(idx), "poseId": str(idx), "intrinsicId": "0", "path": str(img_path), "width": str(w), "height": str(h)}
            ))

        self.logger.info("Found %d images in directory", len(views))
        return views, None

    def write_output_sfm(
        self,
        output_sfm_path: Path,
        original_sfm_dict: Optional[Dict[str, Any]],
        processed_views: List[Tuple[SfmViewItem, Path, Path]],  # item, rgba_path, mask_path
    ) -> Path:
        """
        Write updated SfMData JSON with paths pointing to transparent RGBA PNGs.
        """
        output_sfm_path.parent.mkdir(parents=True, exist_ok=True)
        base_dict: Dict[str, Any] = {}
        if original_sfm_dict and isinstance(original_sfm_dict, dict):
            import copy
            base_dict = copy.deepcopy(original_sfm_dict)
        else:
            base_dict = {
                "version": ["1", "0", "0"],
                "featuresFolders": [],
                "matchesFolders": [],
                "views": [],
                "intrinsics": [],
                "poses": []
            }

        new_views: List[Dict[str, Any]] = []
        for item, rgba_path, mask_path in processed_views:
            v = dict(item.raw_dict) if item.raw_dict else {}
            v["viewId"] = str(item.view_id)
            v["poseId"] = str(item.pose_id)
            v["intrinsicId"] = str(item.intrinsic_id)
            v["path"] = normalize_to_windows_path(rgba_path.resolve())
            v["maskPath"] = normalize_to_windows_path(mask_path.resolve())
            v["width"] = str(item.width)
            v["height"] = str(item.height)
            new_views.append(v)

        base_dict["views"] = new_views

        output_sfm_path.write_text(json.dumps(base_dict, indent=4), encoding="utf-8")
        self.logger.info("Exported updated SfMData JSON to: %s", output_sfm_path)
        return output_sfm_path
