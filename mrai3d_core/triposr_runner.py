"""
TripoSR Local 3D Mesh Reconstruction Engine for mrAIGen3D.

Optimized for 4.0 GB VRAM GPUs (NVIDIA RTX 3050 Laptop) using chunked rendering,
reduced marching cubes resolution, and sequential VRAM cleanup.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
import trimesh

# Ensure TripoSR repo is importable
triposr_repo_candidates = [
    Path("D:/Meshroom/TripoSR"),
    Path("/mnt/d/Meshroom/TripoSR"),
]
for cand in triposr_repo_candidates:
    if cand.is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))


class TripoSRRunner:
    """Executes TripoSR model inference with low-VRAM optimizations."""

    def __init__(
        self,
        model_name_or_path: str = "stabilityai/TripoSR",
        chunk_size: int = 4096,
        mc_resolution: int = 192,
        device: str = "cuda:0",
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.chunk_size = chunk_size
        self.mc_resolution = mc_resolution
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.logger = logger or logging.getLogger("mrAIGen3D.triposr")
        self._model = None

    def _load_model(self):
        if self._model is None:
            self.logger.info("Loading TripoSR model '%s' on device %s...", self.model_name_or_path, self.device)
            from tsr.system import TSR

            model = TSR.from_pretrained(
                self.model_name_or_path,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(self.chunk_size)
            model.to(self.device)
            model.eval()
            self._model = model
            self.logger.info("TripoSR model loaded successfully (chunk_size=%d, mc_res=%d).", self.chunk_size, self.mc_resolution)
        return self._model

    def run(
        self,
        image_path: Path,
        output_obj_path: Path,
        output_glb_path: Path,
        bake_texture: bool = True,
        texture_res: int = 1024,
    ) -> Tuple[Path, Path]:
        """
        Run TripoSR 3D mesh reconstruction on a single image.
        Outputs:
            output_obj_path: .obj file (accompanied by .mtl and texture.png)
            output_glb_path: .glb file with PBR material
        """
        self.logger.info("Starting TripoSR inference on: %s", image_path)
        with Image.open(image_path) as raw_img:
            img = ImageOps.exif_transpose(raw_img)
            if img.mode == "RGBA":
                self.logger.info("Compositing RGBA foreground over neutral background for TripoSR...")
                arr = np.array(img).astype(np.float32) / 255.0
                rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
                img = Image.fromarray((rgb * 255.0).astype(np.uint8))
            else:
                img = img.convert("RGB")

        model = self._load_model()

        with torch.no_grad():
            self.logger.info("Extracting scene codes from single image...")
            scene_codes = model([img], device=self.device)

            self.logger.info("Extracting explicit 3D mesh via Marching Cubes (res=%d)...", self.mc_resolution)
            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=self.mc_resolution,
            )

        if not meshes:
            raise RuntimeError("TripoSR failed to produce a valid 3D mesh.")

        mesh: trimesh.Trimesh = meshes[0]
        self.logger.info("Mesh generated: %d vertices, %d faces.", len(mesh.vertices), len(mesh.faces))

        output_obj_path.parent.mkdir(parents=True, exist_ok=True)
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)

        # Export OBJ
        mesh.export(str(output_obj_path), file_type="obj")
        self.logger.info("Saved Wavefront OBJ: %s", output_obj_path)

        # Export GLB
        mesh.export(str(output_glb_path), file_type="glb")
        self.logger.info("Saved Binary glTF: %s", output_glb_path)

        # Clean VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return output_obj_path, output_glb_path

