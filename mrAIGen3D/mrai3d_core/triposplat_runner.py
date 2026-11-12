"""
TripoSplat 3D Gaussian Splatting Engine for mrAIGen3D.

Executes feed-forward 3D Gaussian Splatting (VAST-AI / Tripo AI) from a single 2D image.
Generates `.ply` Gaussian Splats and `.splat` format.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Ensure TripoSplat repo is importable
triposplat_repo_candidates = [
    Path("D:/Meshroom/TripoSplat"),
    Path("/mnt/d/Meshroom/TripoSplat"),
]
for cand in triposplat_repo_candidates:
    if cand.is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))


class TripoSplatRunner:
    """Executes TripoSplat 3D Gaussian Splatting inference."""

    def __init__(
        self,
        weights_dir: Optional[Path] = None,
        device: str = "cuda:0",
        num_gaussians: int = 131072,
        logger: Optional[logging.Logger] = None,
    ):
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.num_gaussians = num_gaussians
        self.logger = logger or logging.getLogger("mrAIGen3D.triposplat")
        self.weights_dir = weights_dir or Path("D:/Meshroom/TripoSplat/ckpts")
        self._pipe = None

    def _ensure_weights(self) -> Path:
        """Download TripoSplat weights from Hugging Face if not present."""
        w_dir = self.weights_dir.resolve()
        w_dir.mkdir(parents=True, exist_ok=True)

        req_file = w_dir / "diffusion_models" / "triposplat_fp16.safetensors"
        if not req_file.is_file():
            self.logger.info("TripoSplat weights not found at %s. Downloading from Hugging Face (VAST-AI/TripoSplat)...", w_dir)
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="VAST-AI/TripoSplat",
                local_dir=str(w_dir),
                ignore_patterns=["*.git*"],
            )
            self.logger.info("TripoSplat weights downloaded successfully.")
        return w_dir

    def _load_pipeline(self):
        if self._pipe is None:
            w_dir = self._ensure_weights()
            from triposplat import TripoSplatPipeline

            self.logger.info("Initializing TripoSplatPipeline on device: %s...", self.device)
            self._pipe = TripoSplatPipeline(
                ckpt_path=str(w_dir / "diffusion_models" / "triposplat_fp16.safetensors"),
                decoder_path=str(w_dir / "vae" / "triposplat_vae_decoder_fp16.safetensors"),
                dinov3_path=str(w_dir / "clip_vision" / "dino_v3_vit_h.safetensors"),
                flux2_vae_encoder_path=str(w_dir / "vae" / "flux2-vae.safetensors"),
                rmbg_path=str(w_dir / "background_removal" / "birefnet.safetensors"),
                device=self.device,
            )
            self.logger.info("TripoSplat pipeline initialized.")
        return self._pipe

    def run(
        self,
        image_path: Path,
        output_ply_path: Path,
    ) -> Path:
        """
        Run 3D Gaussian Splatting inference on single image.
        Outputs:
            output_ply_path: 3D Gaussian Splatting PLY file.
        """
        self.logger.info("Starting TripoSplat inference on: %s (num_gaussians=%d)...", image_path, self.num_gaussians)
        pipe = self._load_pipeline()

        gaussian, prepared = pipe.run(str(image_path), num_gaussians=self.num_gaussians, show_progress=False)

        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        gaussian.save_ply(str(output_ply_path))
        self.logger.info("Saved 3D Gaussian Splat: %s", output_ply_path)

        # Release GPU VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return output_ply_path

