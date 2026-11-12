"""CLI entry point wrapper for gensfm_core.gensfm_runner."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gensfm_core.gensfm_runner import GenSfMRunnerConfig, GenSfMRunnerResult, main, run_gensfm_pipeline

__all__ = ["GenSfMRunnerConfig", "GenSfMRunnerResult", "run_gensfm_pipeline", "main"]

if __name__ == "__main__":
    main()
