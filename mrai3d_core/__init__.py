"""Core package for mrAIGen3D plugin."""

from .sfm_helper import resolve_single_image
from .cloud_bridge import ColabBridgeClient
from .wsl_bridge import run_ai3d_in_wsl, windows_to_wsl_path, is_wsl_available

__all__ = [
    "resolve_single_image",
    "ColabBridgeClient",
    "run_ai3d_in_wsl",
    "windows_to_wsl_path",
    "is_wsl_available",
    "get_triposr_runner",
    "get_triposplat_runner",
]


def get_triposr_runner():
    """Lazy load TripoSRRunner only in environments with PyTorch & CUDA."""
    from .triposr_runner import TripoSRRunner
    return TripoSRRunner


def get_triposplat_runner():
    """Lazy load TripoSplatRunner only in environments with PyTorch & CUDA."""
    from .triposplat_runner import TripoSplatRunner
    return TripoSplatRunner
