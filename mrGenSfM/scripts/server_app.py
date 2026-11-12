"""
FastAPI Cloud GPU Worker Wrapper (`scripts/server_app.py`) for mrGenSfM.

Re-exports the FastAPI application from `scripts/cloud_worker.py` supporting both
`POST /api/v1/jobs`, `GET /api/v1/jobs/{job_id}`, and `POST /reconstruct`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cloud_worker import app, create_job_endpoint, get_job_status_endpoint, health_check, reconstruct_endpoint

__all__ = [
    "app",
    "health_check",
    "create_job_endpoint",
    "get_job_status_endpoint",
    "reconstruct_endpoint",
]

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
