"""
Zero-Setup Smart Dependency Loader & Environment Bootstrap for mrGenSfM.

Automatically bridges:
- Existing Meshroom 2025.1.0 libraries (numpy, pyalicevision, OpenImageIO)
- Bundled sibling plugin virtual environments (e.g., mrSegmentation/venv/Lib/site-packages)
- Environment variables from .env and meshroom/config.json without hardcoded secrets.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROVIDER_ENV_KEYS: Dict[str, List[str]] = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
    "fal_ai": ["FAL_KEY"],
    "replicate": ["REPLICATE_API_TOKEN"],
    "custom_server": ["GENSFM_API_KEY"],
}


def load_dotenv_file(dotenv_path: Path) -> Dict[str, str]:
    """Parse a standard .env file and populate os.environ for unset keys."""
    loaded: Dict[str, str] = {}
    if not dotenv_path.is_file():
        return loaded

    content = dotenv_path.read_text(encoding="utf-8", errors="replace")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            loaded[key] = value
            if value and not os.environ.get(key):
                os.environ[key] = value
    return loaded


def find_meshroom_root(plugin_root: Path) -> Optional[Path]:
    """Locate the Meshroom-2025.1.0 installation directory."""
    candidates: List[Path] = [
        plugin_root.parent.parent,
        plugin_root.parent / "Meshroom-2025.1.0",
        Path("D:/Meshroom/Meshroom-2025.1.0"),
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if (resolved / "Meshroom.exe").is_file() or (resolved / "aliceVision").is_dir():
                return resolved
        except OSError as exc:
            logging.getLogger(__name__).debug(
                f"Skipping inaccessible Meshroom candidate {candidate}: {exc}"
            )
            continue
    return None


def bootstrap_environment(
    plugin_root: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Configure sys.path, DLL directories, and environment variables for zero-setup execution.
    """
    if plugin_root is None:
        plugin_root = Path(__file__).resolve().parents[1]
    else:
        plugin_root = Path(plugin_root).resolve()

    status: Dict[str, str] = {"plugin_root": str(plugin_root)}

    # 1. Load .env files (plugin root and parent workspace root)
    for env_candidate in (plugin_root / ".env", plugin_root.parent / ".env"):
        if env_candidate.is_file():
            load_dotenv_file(env_candidate)
            status["dotenv"] = str(env_candidate)
            break

    # 2. Load meshroom/config.json defaults if not already set in os.environ
    plugin_cfg = plugin_root / "meshroom" / "config.json"
    if plugin_cfg.is_file():
        try:
            entries = json.loads(plugin_cfg.read_text(encoding="utf-8"))
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict) and "key" in item and "value" in item:
                        k = str(item["key"])
                        v = str(item["value"])
                        if item.get("type") == "path":
                            v = str((plugin_cfg.parent / v).resolve())
                        os.environ.setdefault(k, v)
        except (OSError, json.JSONDecodeError) as exc:
            (logger or logging.getLogger(__name__)).warning(f"Failed to read meshroom/config.json: {exc}")

    # 3. Ensure plugin_root is on sys.path
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

    # 4. Discover Meshroom-2025.1.0 and sibling mrSegmentation venv for Zero-Setup Smart Dependency reuse
    meshroom_root = find_meshroom_root(plugin_root)
    if meshroom_root is not None:
        status["meshroom_root"] = str(meshroom_root)
        av_root = meshroom_root / "aliceVision"
        if av_root.is_dir():
            os.environ.setdefault("ALICEVISION_ROOT", str(av_root))
            status["alicevision_root"] = str(av_root)

        extra_sys_paths: List[Path] = [
            plugin_root / "venv" / "Lib" / "site-packages",
            meshroom_root / "plugins" / "mrSegmentation" / "venv" / "Lib" / "site-packages",
            meshroom_root / "plugins" / "mrSegmentation",
            meshroom_root / "aliceVision" / "lib" / "python",
            meshroom_root / "lib",
        ]
        for extra_path in extra_sys_paths:
            if extra_path.is_dir() and str(extra_path) not in sys.path:
                sys.path.append(str(extra_path))

        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            dll_dirs: List[Path] = [
                meshroom_root / "aliceVision" / "bin",
                meshroom_root / "lib" / "numpy.libs",
                meshroom_root / "lib",
            ]
            for dll_dir in dll_dirs:
                if dll_dir.is_dir():
                    try:
                        os.add_dll_directory(str(dll_dir))
                    except OSError as dll_exc:
                        (logger or logging.getLogger(__name__)).debug(
                            f"Could not register DLL directory {dll_dir}: {dll_exc}"
                        )

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("ATTN_BACKEND_WITH_ATTN_BIAS", "sdpa")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if logger:
        logger.debug(f"Bootstrapped mrGenSfM environment: {status}")

    return status


def resolve_api_key(provider: str, override_key: str = "", plugin_root: Optional[Path] = None) -> str:
    """Retrieve the API key for a given provider from override, environment variables (including GENSFM_API_KEY), or config.json."""
    cleaned_override = override_key.strip()
    if cleaned_override:
        return cleaned_override

    provider_norm = provider.strip().lower()
    env_candidates = list(PROVIDER_ENV_KEYS.get(provider_norm, ["GENSFM_API_KEY"]))
    if "GENSFM_API_KEY" not in env_candidates:
        env_candidates.append("GENSFM_API_KEY")
    for env_name in env_candidates:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val

    if plugin_root is None:
        plugin_root = Path(__file__).resolve().parents[1]
    root_cfg = Path(plugin_root).resolve() / "config.json"
    if root_cfg.is_file():
        try:
            data = json.loads(root_cfg.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key_field in ("cloud_api_key", "api_key", "GENSFM_API_KEY"):
                    cfg_val = str(data.get(key_field, "") or "").strip()
                    if cfg_val:
                        return cfg_val
        except (OSError, json.JSONDecodeError) as cfg_exc:
            logging.getLogger(__name__).debug(f"Could not read API key from {root_cfg}: {cfg_exc}")

    return ""


def resolve_endpoint_url(override_url: str = "", plugin_root: Optional[Path] = None) -> str:
    """Resolve the Cloud REST endpoint URL from node attribute, environment (GENSFM_CLOUD_URL / GENSFM_DEFAULT_ENDPOINT), or config.json."""
    if override_url.strip():
        return override_url.strip().rstrip("/")

    for env_key in ("GENSFM_CLOUD_URL", "GENSFM_DEFAULT_ENDPOINT"):
        env_url = os.environ.get(env_key, "").strip()
        if env_url:
            return env_url.rstrip("/")

    if plugin_root is None:
        plugin_root = Path(__file__).resolve().parents[1]
    root_cfg = plugin_root / "config.json"
    if root_cfg.is_file():
        try:
            data = json.loads(root_cfg.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("default_endpoint"):
                return str(data["default_endpoint"]).strip().rstrip("/")
        except (OSError, json.JSONDecodeError) as cfg_exc:
            logging.getLogger(__name__).debug(f"Could not parse {root_cfg}: {cfg_exc}")

    return "http://127.0.0.1:8000"
