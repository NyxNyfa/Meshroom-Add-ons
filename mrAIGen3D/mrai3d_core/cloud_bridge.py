"""
Colab GPU Bridge & HTTP Client for mrAIGen3D.

Communicates with remote Colab FastAPI server (PRD Section 6) to offload heavy
models (SF3D, SPAR3D, Trellis, Hunyuan3D-2.1) via HTTP Tunnel.
Implemented using standard library urllib so it works in any Python environment
(including Meshroom's embedded cx_Freeze Python) without third-party dependencies.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ColabBridgeClient:
    """HTTP Client for Meshroom AI3D Colab GPU Worker."""

    def __init__(self, endpoint_url: str, timeout: int = 600, logger: Optional[logging.Logger] = None):
        self.endpoint_url = endpoint_url.strip().rstrip("/")
        self.timeout = timeout
        self.logger = logger or logging.getLogger("mrAIGen3D.cloud_bridge")

    def check_health(self) -> Dict[str, Any]:
        """Check status of remote Colab GPU worker."""
        url = f"{self.endpoint_url}/health"
        self.logger.info("Checking Colab GPU worker health at: %s", url)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Meshroom-mrAIGen3D"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.logger.info("Colab worker online: GPU=%s, VRAM=%s MB", data.get("gpu"), data.get("vram_total_mb"))
                return data
        except Exception as e:
            self.logger.error("Failed to connect to Colab worker: %s", e)
            raise ConnectionError(f"Could not reach Colab worker at {url}: {e}") from e

    def generate(
        self,
        image_path: Path,
        output_dir: Path,
        model_name: str = "spar3d",
        remove_bg: bool = True,
        texture_res: int = 1024,
        low_vram: bool = True,
        hunyuan_skip_paint: bool = False,
        hf_token: str = "",
    ) -> Tuple[Path, Path]:
        """
        Send generation request to Colab worker and download resultant 3D zip archive.
        Returns:
            Tuple of (output_obj_path, output_glb_path)
        """
        url = f"{self.endpoint_url}/generate"
        self.logger.info("Submitting job to Colab worker [%s] on model '%s'...", url, model_name)

        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build multipart/form-data payload with pure standard library
        boundary = f"----MeshroomBoundary{uuid.uuid4().hex}"
        body = bytearray()

        fields = {
            "model_name": model_name,
            "remove_bg": "true" if remove_bg else "false",
            "texture_res": str(texture_res),
            "low_vram": "true" if low_vram else "false",
            "hunyuan_skip_paint": "true" if hunyuan_skip_paint else "false",
            "hf_token": hf_token or "",
        }

        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(f"{value}\r\n".encode("utf-8"))

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode("utf-8")
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        with open(image_path, "rb") as f_img:
            body.extend(f_img.read())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Meshroom-mrAIGen3D",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read()
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            self.logger.error("Colab worker returned HTTP error %d: %s", e.code, err_msg)
            raise RuntimeError(f"Colab worker error {e.code}: {err_msg}") from e
        except Exception as e:
            self.logger.error("Failed to connect or submit job to Colab worker: %s", e)
            raise ConnectionError(f"Could not reach Colab worker at {url}: {e}") from e

        # Unpack returned zip archive
        self.logger.info("Generation successful. Extracting returned 3D archive...")
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(output_dir)

        obj_candidates = list(output_dir.glob("*.obj"))
        glb_candidates = list(output_dir.glob("*.glb"))

        obj_path = obj_candidates[0] if obj_candidates else output_dir / "mesh.obj"
        glb_path = glb_candidates[0] if glb_candidates else output_dir / "asset_pbr.glb"

        self.logger.info("Downloaded 3D assets to: %s", output_dir)
        return obj_path, glb_path
