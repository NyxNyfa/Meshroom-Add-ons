"""Core package for mrRemBG plugin."""

from .sfm_io import SfmDataIO, SfmViewItem, read_image_dimensions
from .wsl_bridge import run_rembg_in_wsl, windows_to_wsl_path, is_wsl_available

__all__ = [
    "SfmDataIO",
    "SfmViewItem",
    "read_image_dimensions",
    "run_rembg_in_wsl",
    "windows_to_wsl_path",
    "is_wsl_available",
    "get_rembg_engine",
]


def get_rembg_engine():
    """Lazy loader for RemBGEngine to avoid importing PIL/torch on host environments."""
    from .engine import RemBGEngine
    return RemBGEngine
