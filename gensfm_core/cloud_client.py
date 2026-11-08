"""
3D Cloud AI & Serverless GPU Client for mrGenSfM.

Supports dispatching 2-4 masked RGBA views to:
- Custom FastAPI GenSfM Worker (`scripts/server_app.py`)
- Hugging Face Inference Endpoints / Spaces
- fal.ai Serverless 3D Endpoints
- Replicate Serverless GPU Endpoints
Downloads and unpacks `mesh.glb`, `gaussian.ply`, `cameras.json`, and `reprojection-*.png`
into `{nodeCacheFolder}`.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests


def _resolve_artifact_url(base_url: str, artifact_url: str) -> str:
    cleaned = artifact_url.strip()
    if cleaned.startswith(("http://", "https://")):
        return cleaned
    return f"{base_url.rstrip('/')}/{cleaned.lstrip('/')}"


def _cleanup_partial_cloud_files(output_dir: Path, logger: logging.Logger) -> None:
    """Clean up partial or temporary download files in output_dir upon Cloud API failure (PRD Section 7.1)."""
    for candidate_name in ("_cloud_partial.zip", "mesh.glb.part", "gaussian.ply.part", "cameras.json.part"):
        part_file = output_dir / candidate_name
        if part_file.is_file():
            try:
                part_file.unlink()
            except OSError as exc:
                logger.debug(f"Could not remove temporary file {part_file}: {exc}")


def _unpack_cloud_response(
    response: requests.Response,
    output_dir: Path,
    headers: Dict[str, str],
    base_url: str = "",
) -> Path:
    """Unpack either a ZIP bundle or JSON files payload into output_dir."""
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/zip" in content_type or response.content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zf:
            for member in zf.namelist():
                safe_name = Path(member).name
                if not safe_name:
                    continue
                (output_dir / safe_name).write_bytes(zf.read(member))
        return output_dir

    payload: Dict[str, Any] = response.json()
    if "artifact_url" in payload and isinstance(payload["artifact_url"], str) and payload["artifact_url"].strip():
        resolved_art = _resolve_artifact_url(base_url, str(payload["artifact_url"]))
        dl_resp = requests.get(resolved_art, headers=headers, timeout=120)
        dl_resp.raise_for_status()
        if (
            "application/zip" in dl_resp.headers.get("Content-Type", "").lower()
            or dl_resp.content[:4] == b"PK\x03\x04"
        ):
            return _unpack_cloud_response(dl_resp, output_dir, headers, base_url=base_url)

    if "files" in payload and isinstance(payload["files"], dict):
        for fname, b64_or_url in payload["files"].items():
            safe_name = Path(fname).name
            dest = output_dir / safe_name
            if isinstance(b64_or_url, str) and (
                b64_or_url.startswith(("http://", "https://")) or b64_or_url.startswith("/")
            ):
                dl = requests.get(_resolve_artifact_url(base_url, b64_or_url), headers=headers, timeout=120)
                dl.raise_for_status()
                dest.write_bytes(dl.content)
            elif isinstance(b64_or_url, str):
                dest.write_bytes(base64.b64decode(b64_or_url))
    if "cameras" in payload and isinstance(payload["cameras"], dict):
        (output_dir / "cameras.json").write_text(
            json.dumps(payload["cameras"], indent=2) + "\n", encoding="utf-8"
        )
    return output_dir


def run_gensfm_cloud_api(
    image_paths: Sequence[Path],
    output_dir: Path,
    endpoint_url: str,
    api_key: str = "",
    provider: str = "custom_server",
    model_name: str = "dotblue/GenSfm",
    seed: int = 1,
    steps: int = 25,
    remove_background: bool = False,
    timeout_sec: int = 300,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Send 2-4 random views to a Cloud GenSfM endpoint (`POST /api/v1/jobs` with status polling
    `GET /api/v1/jobs/{job_id}`, or synchronous `POST /reconstruct`) and unpack the resulting 3D
    bundle into output_dir.
    """
    if len(image_paths) < 2:
        raise ValueError("At least 2 images are required for Cloud GenSfM reconstruction.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_url = endpoint_url.strip().rstrip("/")
    if not clean_url:
        raise ValueError("Cloud endpoint URL cannot be empty.")

    headers: Dict[str, str] = {}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
        headers["X-API-Key"] = api_key.strip()

    provider_norm = provider.strip().lower()
    if logger:
        logger.info(
            f"Sending {len(image_paths)} views to Cloud 3D API ({provider_norm}) at {clean_url}..."
        )

    params_json = json.dumps(
        {
            "seed": int(seed),
            "steps": int(steps),
            "remove_background": bool(remove_background),
            "pretrained": model_name,
        }
    )

    active_logger = logger or logging.getLogger(__name__)
    files_payload: List[Any] = []
    opened_files = []
    try:
        for idx, img_path in enumerate(image_paths):
            fh = open(img_path, "rb")
            opened_files.append(fh)
            files_payload.append(("images", (f"input-rgba-{idx:02d}.png", fh, "image/png")))

        form_data = {
            "seed": str(int(seed)),
            "steps": str(int(steps)),
            "remove_background": "true" if remove_background else "false",
            "pretrained": model_name,
            "provider": provider_norm,
            "parameters": params_json,
        }

        if clean_url.endswith(("/api/v1/jobs", "/reconstruct")):
            target_endpoint = clean_url
            base_server_url = clean_url.rsplit("/api/v1/jobs", 1)[0].rsplit("/reconstruct", 1)[0]
        elif provider_norm == "custom_server":
            target_endpoint = f"{clean_url}/api/v1/jobs"
            base_server_url = clean_url
        else:
            target_endpoint = f"{clean_url}/api/v1/jobs"
            base_server_url = clean_url

        response = requests.post(
            target_endpoint,
            headers=headers,
            files=files_payload,
            data=form_data,
            timeout=timeout_sec,
        )
        if response.status_code == 404 and target_endpoint.endswith("/api/v1/jobs"):
            # Fallback to /reconstruct for legacy servers
            for fh in opened_files:
                fh.seek(0)
            target_endpoint = f"{base_server_url}/reconstruct"
            response = requests.post(
                target_endpoint,
                headers=headers,
                files=files_payload,
                data=form_data,
                timeout=timeout_sec,
            )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/zip" in content_type or response.content[:4] == b"PK\x03\x04":
            return _unpack_cloud_response(response, output_dir, headers, base_url=base_server_url)

        payload: Dict[str, Any] = response.json()
        # Handle asynchronous job polling contract: GET {cloudEndpoint}/api/v1/jobs/{job_id}
        has_inline_artifacts = bool(payload.get("files") or payload.get("cameras") or payload.get("artifact_url"))
        if "job_id" in payload and (payload.get("status") != "completed" or not has_inline_artifacts):
            job_id = str(payload["job_id"])
            status_url = f"{base_server_url}/api/v1/jobs/{job_id}"
            deadline = time.monotonic() + float(max(10, timeout_sec))
            while time.monotonic() < deadline:
                poll_resp = requests.get(status_url, headers=headers, timeout=30)
                poll_resp.raise_for_status()
                if (
                    "application/zip" in poll_resp.headers.get("Content-Type", "").lower()
                    or poll_resp.content[:4] == b"PK\x03\x04"
                ):
                    return _unpack_cloud_response(poll_resp, output_dir, headers, base_url=base_server_url)
                poll_data = poll_resp.json()
                status_val = str(poll_data.get("status", ""))
                stage_val = str(poll_data.get("stage", "processing"))
                prog_val = poll_data.get("progress", 0)
                active_logger.info(
                    f"[Cloud Job {job_id}] status={status_val} stage={stage_val} progress={prog_val}%"
                )
                if status_val == "completed":
                    if "artifact_url" in poll_data and str(poll_data["artifact_url"]).strip():
                        art_url = _resolve_artifact_url(base_server_url, str(poll_data["artifact_url"]))
                        art_resp = requests.get(art_url, headers=headers, timeout=120)
                        art_resp.raise_for_status()
                        return _unpack_cloud_response(art_resp, output_dir, headers, base_url=base_server_url)
                    return _unpack_cloud_response(poll_resp, output_dir, headers, base_url=base_server_url)
                if status_val in ("failed", "error"):
                    raise RuntimeError(
                        f"Cloud GenSfM job {job_id} failed: {poll_data.get('error', 'unknown error')}"
                    )
                time.sleep(1.0)
            raise TimeoutError(f"Cloud GenSfM job {job_id} timed out after {timeout_sec}s.")

        return _unpack_cloud_response(response, output_dir, headers, base_url=base_server_url)
    except requests.exceptions.HTTPError as http_exc:
        status_code = getattr(http_exc.response, "status_code", "unknown")
        active_logger.error(
            f"Cloud API HTTP error (status={status_code}) at {clean_url}: {http_exc}"
        )
        _cleanup_partial_cloud_files(output_dir, active_logger)
        raise
    except requests.exceptions.RequestException as req_exc:
        active_logger.error(f"Cloud API connection/timeout error at {clean_url}: {req_exc}")
        _cleanup_partial_cloud_files(output_dir, active_logger)
        raise
    finally:
        for fh in opened_files:
            try:
                fh.close()
            except OSError as close_exc:
                active_logger.debug(
                    f"Failed to close uploaded image handle: {close_exc}"
                )
