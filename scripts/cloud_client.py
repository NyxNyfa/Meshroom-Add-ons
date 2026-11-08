"""CLI & module wrapper for gensfm_core.cloud_client."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gensfm_core.cloud_client import run_gensfm_cloud_api

__all__ = ["run_gensfm_cloud_api"]
