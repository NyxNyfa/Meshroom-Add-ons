"""
FastAPI Cloud GPU Worker Reference (`scripts/cloud_worker.py`) for mrGenSfM.

Implements the Cloud AI API Contract specified in PRD Section 5.1:
- `POST /api/v1/jobs`: Accepts 2-4 multipart images (`images`) and JSON/form parameters
  (`seed`, `steps`, `remove_background`), authenticates `Authorization: Bearer <token>` or `X-API-Key`,
  and returns either the completed artifact bundle or a job status object.
- `GET /api/v1/jobs/{job_id}`: Returns job status (`status`, `stage`, `progress`) or completed artifact bundle.
- `POST /reconstruct`: Synchronous ZIP bundle endpoint for direct reconstruction.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile

from gensfm_core.env_bootstrap import bootstrap_environment
from gensfm_core.gensfm_runner import GenSfMRunnerConfig, run_gensfm_pipeline

app = FastAPI(
    title="mrGenSfM Cloud AI Worker",
    version="1.0.0",
    description="Reference Cloud GPU Server for AliceVision Meshroom GenSfM sparse 2-4 random-view reconstruction.",
)

_COMPLETED_JOBS: Dict[str, bytes] = {}


def _verify_auth(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    expected_key = os.environ.get("GENSFM_API_KEY", "").strip()
    if not expected_key:
        return
    bearer_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer_token = authorization[7:].strip()
    token = bearer_token or (x_api_key or "").strip()
    if token != expected_key:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid Bearer token or X-API-Key.")


def _execute_reconstruction_bundle(
    image_payloads: List[tuple[str, bytes]],
    seed: int,
    steps: int,
    remove_background: bool,
    pretrained: str,
) -> bytes:
    bootstrap_environment()
    with tempfile.TemporaryDirectory(prefix="gensfm_cloud_job_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        in_dir = tmp_path / "inputs"
        out_dir = tmp_path / "outputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (filename, content) in enumerate(image_payloads):
            suffix = Path(filename or f"{idx:02d}.png").suffix or ".png"
            dest = in_dir / f"{idx:02d}{suffix}"
            dest.write_bytes(content)

        cfg = GenSfMRunnerConfig(
            input_path=str(in_dir),
            output_dir=str(out_dir),
            engine_type="gensfm_diffusion",
            execution_mode="local_venv",
            masking_mode="auto_rembg" if remove_background else "existing_alpha",
            seed=int(seed),
            steps=int(steps),
            pretrained_model=pretrained,
            export_alicevision_sfm=False,
        )
        run_gensfm_pipeline(cfg)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in out_dir.iterdir():
                if item.is_file() and not item.name.startswith("_"):
                    zf.write(item, arcname=item.name)
        return zip_buf.getvalue()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "mrGenSfM Cloud Worker"}


@app.post("/api/v1/jobs")
async def create_job_endpoint(
    images: Optional[List[UploadFile]] = File(default=None),
    images_brackets: Optional[List[UploadFile]] = File(default=None, alias="images[]"),
    parameters: Optional[str] = Form(default=None),
    seed: int = Form(1),
    steps: int = Form(25),
    remove_background: bool = Form(False),
    pretrained: str = Form("dotblue/GenSfm"),
    async_poll: bool = Form(False),
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    accept: Optional[str] = Header(default=None),
) -> Response:
    _verify_auth(authorization, x_api_key)
    upload_list = list(images or []) + list(images_brackets or [])
    if len(upload_list) < 2 or len(upload_list) > 4:
        raise HTTPException(
            status_code=400,
            detail="GenSfM requires at least 2 views of the object (4 random views recommended).",
        )

    if parameters:
        try:
            parsed = json.loads(parameters)
            seed = int(parsed.get("seed", seed))
            steps = int(parsed.get("steps", steps))
            remove_background = bool(parsed.get("remove_background", remove_background))
            pretrained = str(parsed.get("pretrained", pretrained))
        except (ValueError, TypeError) as param_exc:
            raise HTTPException(status_code=400, detail=f"Invalid parameters JSON: {param_exc}") from param_exc

    img_blobs: List[tuple[str, bytes]] = []
    for idx, upload in enumerate(upload_list):
        img_blobs.append((upload.filename or f"{idx:02d}.png", await upload.read()))

    bundle_bytes = _execute_reconstruction_bundle(
        image_payloads=img_blobs,
        seed=seed,
        steps=steps,
        remove_background=remove_background,
        pretrained=pretrained,
    )
    job_id = uuid.uuid4().hex
    _COMPLETED_JOBS[job_id] = bundle_bytes

    if async_poll or (accept and "application/json" in accept.lower()):
        return Response(
            content=json.dumps(
                {
                    "job_id": job_id,
                    "status": "completed",
                    "stage": "completed",
                    "progress": 100,
                    "artifact_url": f"/api/v1/jobs/{job_id}/artifacts",
                }
            ),
            media_type="application/json",
            headers={"X-Job-Id": job_id},
        )

    return Response(
        content=bundle_bytes,
        media_type="application/zip",
        headers={
            "X-Job-Id": job_id,
            "Content-Disposition": f'attachment; filename="gensfm_{job_id}.zip"',
        },
    )


@app.get("/api/v1/jobs/{job_id}")
def get_job_status_endpoint(
    job_id: str,
    format: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    accept: Optional[str] = Header(default=None),
) -> Response:
    _verify_auth(authorization, x_api_key)
    if job_id not in _COMPLETED_JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if format == "zip" or (accept and "application/zip" in accept.lower()):
        bundle = _COMPLETED_JOBS[job_id]
        return Response(
            content=bundle,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="gensfm_{job_id}.zip"'},
        )
    return Response(
        content=json.dumps(
            {
                "job_id": job_id,
                "status": "completed",
                "stage": "completed",
                "progress": 100,
                "artifact_url": f"/api/v1/jobs/{job_id}/artifacts",
            }
        ),
        media_type="application/json",
    )


@app.get("/api/v1/jobs/{job_id}/artifacts")
def get_job_artifacts_endpoint(
    job_id: str,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> Response:
    _verify_auth(authorization, x_api_key)
    if job_id not in _COMPLETED_JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    bundle = _COMPLETED_JOBS[job_id]
    return Response(
        content=bundle,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="gensfm_{job_id}.zip"'},
    )


@app.post("/reconstruct")
async def reconstruct_endpoint(
    images: List[UploadFile] = File(...),
    seed: int = Form(1),
    steps: int = Form(25),
    remove_background: bool = Form(False),
    pretrained: str = Form("dotblue/GenSfm"),
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> Response:
    return await create_job_endpoint(
        images=images,
        parameters=None,
        seed=seed,
        steps=steps,
        remove_background=remove_background,
        pretrained=pretrained,
        authorization=authorization,
        x_api_key=x_api_key,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

