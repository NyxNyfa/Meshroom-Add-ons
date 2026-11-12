__version__ = "1.0"

import os
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class AI3DGenerator(desc.Node):
    """
    Single-Image Generative 3D Mesh & Gaussian Splatting Node (`AI3DGenerator`).
    Powered by Stability AI & VAST-AI Research.

    Transforms a single 2D image (from mrRemBG or CameraInit) into a textured 3D mesh
    (.obj and .glb) or a 3D Gaussian Splat (.ply).
    Optimized for laptops with 4.0 GB VRAM (RTX 3050 Laptop) via chunked rendering,
    and bridges heavy models to Google Colab.
    """

    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("input")
    category = "AI Generation / 3D Reconstruction"
    documentation = """
Single-Image Generative 3D Mesh & Gaussian Splatting Node (`AI3DGenerator`) for AliceVision Meshroom.

Powered by Stability AI.
This Stability AI Model is licensed under the Stability AI Community License,
Copyright © Stability AI Ltd. All Rights Reserved.

### Overview
Transforms a single 2D image into 3D geometry in seconds:
- **TripoSR**: Feed-forward 3D mesh generator producing high quality Wavefront `.obj` and Binary `.glb` (PBR).
- **TripoSplat**: Feed-forward 3D Gaussian Splatting producing `.ply` point clouds.
- **Remote Cloud Models (via Colab Bridge)**: Stable-Fast-3D, SPAR3D, TRELLIS, and Hunyuan3D-2.1.

### Recommended Pipeline Topology
Connect `CameraInit` -> `RemBG` -> `AI3DGenerator` -> `MeshFiltering` -> `MeshDecimate` -> `Publish`.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData (.sfm) from RemBG / CameraInit, or directory of images, or direct single image file.",
            value="",
        ),
        desc.IntParam(
            name="selectedViewIndex",
            label="Selected View Index",
            description="Index of the image (0-based) to use when input contains multiple photos.",
            value=0,
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="modelName",
            label="AI 3D Model",
            description="Generative 3D model architecture:\n"
                        " - triposr: Fast local 3D mesh (RTX 3050 4GB / CPU) [Stability AI / Tripo]\n"
                        " - triposplat: Fast local 3D Gaussian Splatting (.ply) [VAST-AI / Tripo]\n"
                        " - stable-fast-3d: Fast UV-unwrapped mesh (Colab recommended)\n"
                        " - spar3d: High detail point-aware 3D (Colab recommended)\n"
                        " - trellis-large: High-fidelity structured latent flow (Colab T4)\n"
                        " - trellis-2-4b: Next-gen 4B parameter model (Colab L4/A100)\n"
                        " - hunyuan3d-2.1: SOTA Shape & Paint 2-stage model (Colab T4)",
            value="triposr",
            values=[
                "triposr",
                "triposplat",
                "stable-fast-3d",
                "spar3d",
                "trellis-large",
                "trellis-2-4b",
                "hunyuan3d-2.1",
            ],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="executionBackend",
            label="Execution Backend",
            description="Execution backend:\n"
                        " - local_wsl_gpu: Local RTX 3050 Laptop GPU via WSL2 (CUDA 12.8).\n"
                        " - local_cpu: Force CPU computation.\n"
                        " - colab_remote: Remote GPU worker running in Google Colab via HTTP tunnel.",
            value="local_wsl_gpu",
            values=["local_wsl_gpu", "local_cpu", "colab_remote"],
            exclusive=True,
        ),
        desc.StringParam(
            name="remoteEndpointUrl",
            label="Colab Remote URL",
            description="Public URL from Google Colab worker (e.g., https://xxxx.trycloudflare.com or gradio.live).",
            value="",
        ),
        desc.ChoiceParam(
            name="mcResolution",
            label="Marching Cubes Resolution",
            description="Grid resolution for TripoSR isosurface extraction (192 = 4GB VRAM safe, 256 = standard).",
            value="192",
            values=["192", "256", "384"],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="chunkSize",
            label="Chunk Size",
            description="Evaluation chunk size for TripoSR NeRF renderer (lower = less VRAM).",
            value="4096",
            values=["2048", "4096", "8192"],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="numGaussians",
            label="Gaussian Count",
            description="Point count for TripoSplat 3D Gaussian Splatting.",
            value="131072",
            values=["32768", "65536", "131072", "262144"],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="textureResolution",
            label="Texture Resolution",
            description="Texture map resolution in pixels.",
            value="1024",
            values=["512", "1024", "2048"],
            exclusive=True,
        ),
        desc.BoolParam(
            name="lowVramMode",
            label="Low VRAM Mode",
            description="Enables sequential stage unloading and CPU offloading.",
            value=True,
        ),
        desc.BoolParam(
            name="hunyuanSkipPaint",
            label="Hunyuan Skip Paint",
            description="For hunyuan3d-2.1: skip texture paint stage (Shape DiT only, saves 21 GB VRAM).",
            value=False,
        ),
        desc.StringParam(
            name="hfToken",
            label="Hugging Face Token",
            description="Access token (hf_...) for gated models (SF3D, SPAR3D).",
            value="",
        ),
        desc.StringParam(
            name="wslDistro",
            label="WSL Distro",
            description="WSL2 distribution name.",
            value="Ubuntu",
        ),
        desc.StringParam(
            name="wslCondaEnv",
            label="WSL Conda Env",
            description="WSL2 Conda environment name.",
            value="gensfm",
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level of execution logs.",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="outputMeshObj",
            label="Output Mesh (.obj)",
            description="Generated 3D mesh in Wavefront OBJ format. Displayed in Meshroom 3D Viewport.",
            semantic="3d",
            value="{nodeCacheFolder}/mesh.obj",
        ),
        desc.File(
            name="outputMeshGlb",
            label="Output Mesh (.glb)",
            description="Generated 3D asset in Binary glTF format with full PBR materials.",
            value="{nodeCacheFolder}/asset_pbr.glb",
        ),
        desc.File(
            name="outputGaussianPly",
            label="Output Gaussian (.ply)",
            description="3D Gaussian Splatting point cloud file for TripoSplat / Trellis.",
            semantic="3d",
            value="{nodeCacheFolder}/gaussian.ply",
        ),
        desc.File(
            name="outputPreviewPng",
            label="Input Preview",
            description="Pre-processed single input image displayed in Meshroom 2D Image Viewer.",
            semantic="image",
            value="{nodeCacheFolder}/preview.png",
        ),
    ]

    def processChunk(self, chunk):
        import shutil
        import sys

        chunk.logManager.start(chunk.node.verboseLevel.value)
        logger = chunk.logger

        plugin_root = Path(__file__).resolve().parents[2]
        if str(plugin_root) not in sys.path:
            sys.path.insert(0, str(plugin_root))

        from mrai3d_core.sfm_helper import resolve_single_image
        from mrai3d_core.wsl_bridge import run_ai3d_in_wsl, is_wsl_available

        input_val = str(chunk.node.input.value or "").strip()
        if not input_val:
            raise ValueError("mrAIGen3D requires an 'input' (SfMData from RemBG / CameraInit, or direct image file).")

        selected_idx = int(chunk.node.selectedViewIndex.value)

        img_path, view_id = resolve_single_image(
            input_sfm_str=input_val,
            selected_view_index=selected_idx,
            logger=logger,
        )

        # Smart Non-Blocking Transparency Validation
        try:
            with open(img_path, "rb") as f_img:
                hdr = f_img.read(32)
                # PNG: color_type is byte 25 (6 = RGBA, 4 = Gray+Alpha)
                if hdr.startswith(b"\x89PNG\r\n\x1a\n") and len(hdr) >= 26:
                    color_type = hdr[25]
                    if color_type not in (4, 6):
                        logger.info(
                            "[AI3DGenerator Tip] Input image does not contain transparency. "
                            "For best 3D results, consider connecting through a 'RemBG' node first."
                        )
                elif not img_path.suffix.lower() == ".png":
                    logger.info(
                        "[AI3DGenerator Tip] Input image format is '%s'. "
                        "Transparent PNG images (e.g. from RemBG) yield the cleanest 3D meshes.",
                        img_path.suffix
                    )
        except Exception:
            pass

        output_obj = Path(chunk.node.outputMeshObj.value).resolve()
        output_glb = Path(chunk.node.outputMeshGlb.value).resolve()
        output_ply = Path(chunk.node.outputGaussianPly.value).resolve()
        preview_png = Path(chunk.node.outputPreviewPng.value).resolve()

        output_obj.parent.mkdir(parents=True, exist_ok=True)

        # Copy selected image to preview
        try:
            shutil.copyfile(img_path, preview_png)
        except Exception:
            pass

        model_name = str(chunk.node.modelName.value)
        backend = str(chunk.node.executionBackend.value)
        remote_url = str(chunk.node.remoteEndpointUrl.value or "").strip()

        if backend == "colab_remote" or model_name not in ("triposr", "triposplat"):
            if not remote_url:
                raise ValueError(
                    f"Model '{model_name}' requires executionBackend='colab_remote' and a valid 'remoteEndpointUrl'."
                )
            logger.info("Executing generation on Colab GPU Worker: %s", remote_url)
            from mrai3d_core.cloud_bridge import ColabBridgeClient
            client = ColabBridgeClient(endpoint_url=remote_url, logger=logger)
            client.generate(
                image_path=img_path,
                output_dir=output_obj.parent,
                model_name=model_name,
                texture_res=int(chunk.node.textureResolution.value),
                low_vram=bool(chunk.node.lowVramMode.value),
                hunyuan_skip_paint=bool(chunk.node.hunyuanSkipPaint.value),
                hf_token=str(chunk.node.hfToken.value or ""),
            )
        else:
            # Local models (TripoSR or TripoSplat)
            logger.info("Executing model '%s' locally via WSL2 (RTX 3050 4GB GPU)...", model_name)
            run_ai3d_in_wsl(
                image_path=img_path,
                output_obj=output_obj,
                output_glb=output_glb,
                output_ply=output_ply,
                model_name=model_name,
                mc_resolution=int(chunk.node.mcResolution.value),
                chunk_size=int(chunk.node.chunkSize.value),
                num_gaussians=int(chunk.node.numGaussians.value),
                distro=str(chunk.node.wslDistro.value),
                conda_env=str(chunk.node.wslCondaEnv.value),
                logger=logger,
            )

        logger.info("AI3DGenerator chunk processing completed successfully.")

