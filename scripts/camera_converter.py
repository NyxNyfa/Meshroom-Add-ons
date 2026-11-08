"""CLI & module wrapper for gensfm_core.camera_converter."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gensfm_core.camera_converter import (
    DEFAULT_DATA_TO_MESH,
    convert_cameras_json_to_sfm,
    gensfm_extrinsic_to_alicevision_pose,
    normalized_intrinsic_to_alicevision,
)

__all__ = [
    "DEFAULT_DATA_TO_MESH",
    "convert_cameras_json_to_sfm",
    "gensfm_extrinsic_to_alicevision_pose",
    "normalized_intrinsic_to_alicevision",
]
