__version__ = "1.0"

import os
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class GenSfM(desc.Node):
    """
    Sparse Random-View AI Initialization & 3D Reconstruction Node (`GenSfM`)
    for AliceVision Meshroom 2025.1.0+.

    Transforms Meshroom's input requirement from 30-150+ sequential high-overlap photos
    down to just 2 to 4 uncalibrated sparse photos from random viewpoints.
    Simultaneously recovers 3D camera poses & intrinsics (`sfmData.abc` / `cameras.json`),
    complete 3D textured geometry (`mesh.obj` / `mesh.glb`), 3D Gaussian Splats (`gaussian.ply`),
    and optional dense synthetic multi-view renders (`sfmDataDense.abc`) for downstream AliceVision nodes.
    """

    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("input")
    category = "Sparse Reconstruction / AI"
    documentation = """
Sparse Random-View AI Initialization & 3D Reconstruction Node (`GenSfM`) for AliceVision Meshroom.

### Transforming Meshroom Input to 2–4 Random Photos (Dual-Branch Hybrid Architecture)
Standard Meshroom pipelines require 30–150+ sequential photos with >60% overlap. With `GenSfM`, you only need **2 to 4 uncalibrated photos of an object from random angles**:
1. Drop 2 to 4 random photos of an object into **Meshroom** (`CameraInit`).
2. Connect `CameraInit.output` to `GenSfM.input` (or open the **`genSfmHybrid`** pipeline template).
3. `GenSfM` simultaneously feeds two active branches on the same canvas:
   - **Fast Branch (Direct AI Polish)**: Connects `outputSfMData` (`sfmData.abc`) and `outputMeshObj` (`mesh.obj`) directly to `PrepareDenseScene_Fast` + `MeshFiltering_Fast` -> `Texturing_Fast` for rapid multi-band 4K texturing.
   - **Dense Branch (Sparse-to-Dense MVS Photogrammetry)**: With `generateSyntheticViews=24` (default), `GenSfM` renders 24 orbital 360-deg multi-elevation synthetic views into `outputSfMDataDense` (`sfmDataDense.abc`) to feed classic AliceVision MVS (`PrepareDenseScene_Dense` -> `DepthMap_Dense` -> `DepthMapFilter_Dense` -> `Meshing_Dense` -> `MeshFiltering_Dense` -> `Texturing_Dense`).
4. **How to View 3D Results in Meshroom Viewport**:
   - Meshroom does not auto-load 3D meshes into GPU memory on project open. **Double-click** `GenSfM_1`, `Texturing_Fast`, or `Texturing_Dense` in the Graph Editor (or click the **Eye icon** on the node) to display the textured 3D mesh and camera frustums in the **3D Viewer**.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="Path to sfmData (.sfm / .abc) from CameraInit or directory containing 2-4 random object photos.",
            value="",
        ),
        desc.File(
            name="inputMasks",
            label="Input Masks",
            description="Optional path to RGBA/grayscale masks folder from mrSegmentation when maskingStrategy='external_masks'.",
            value="",
        ),
        desc.ChoiceParam(
            name="maskingStrategy",
            label="Masking Strategy",
            description="Foreground segmentation strategy for isolating the target object (RGBA):\n"
                        " - auto_rembg: Automatic background removal (--remove-background rembg / smart fallback).\n"
                        " - existing_alpha: Use existing PNG alpha channel.\n"
                        " - external_masks: Combine RGB from CameraInit with masks from inputMasks (mrSegmentation).\n"
                        " - auto_birefnet: High-precision local matting reusing BiRefNet from mrSegmentation if installed.\n"
                        " - auto_grabcut: Fast OpenCV center-object GrabCut segmentation.",
            value="auto_rembg",
            values=[
                "auto_rembg",
                "existing_alpha",
                "external_masks",
                "auto_birefnet",
                "auto_grabcut",
            ],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="viewSelectionMode",
            label="View Selection Mode",
            description="How to select 2 to 4 photos from CameraInit:\n"
                        " - all_up_to_4: Take the first 2-4 photos sequentially.\n"
                        " - manual_indices: Use the comma-separated indices in Selected Indices.\n"
                        " - auto_diverse: Automatically pick the 4 most visually diverse & sharp views.\n"
                        " - uniform_spread: Evenly space 4 views across the dataset.\n"
                        " - ai_smart_select: Alias for feature-based diverse view selection.",
            value="all_up_to_4",
            values=[
                "all_up_to_4",
                "manual_indices",
                "auto_diverse",
                "uniform_spread",
                "ai_smart_select",
            ],
            exclusive=True,
        ),
        desc.StringParam(
            name="selectedIndices",
            label="Selected Indices",
            description="Comma-separated 0-based photo indices when viewSelectionMode='manual_indices' (e.g. '0,1,2,3').",
            value="0,1,2,3",
        ),
        desc.ChoiceParam(
            name="executionBackend",
            label="Execution Backend",
            description="Hardware execution backend for GenSfM:\n"
                        " - local_wsl_gpu: Execute natively on local NVIDIA GPU inside WSL2 Ubuntu (with Low-VRAM offloading).\n"
                        " - local_docker: Execute inside Linux CUDA 12.8 Docker container (WSL2 backend).\n"
                        " - local_venv: Execute directly in local Python virtual environment.",
            value="local_wsl_gpu",
            values=["local_wsl_gpu", "local_docker", "local_venv", "cloud_api"],
            exclusive=True,
            invalidate=False,
        ),
        desc.ChoiceParam(
            name="engineType",
            label="Reconstruction Engine",
            description="Select the 3D reconstruction engine:\n"
                        " - gensfm_diffusion: Native 3D Diffusion (dotb1ue/GenSfm) for joint 3D shape, appearance & pose.",
            value="gensfm_diffusion",
            values=["gensfm_diffusion", "llm_guided_sfm"],
            exclusive=True,
            advanced=True,
        ),
        desc.ChoiceParam(
            name="aiProvider",
            label="AI / Cloud Provider (Advanced)",
            description="AI API provider (optional/advanced).",
            value="gemini",
            values=[
                "gemini",
                "openai",
                "anthropic",
                "custom_server",
                "huggingface",
                "fal_ai",
                "replicate",
            ],
            exclusive=True,
            advanced=True,
            invalidate=False,
        ),
        desc.StringParam(
            name="aiModelName",
            label="AI Model Name (Advanced)",
            description="Model identifier for legacy provider.",
            value="gemini-3.8-flash",
            advanced=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="lowVramMode",
            label="Low-VRAM Mode (4GB-8GB GPU)",
            description="Enable FP16 precision, expandable CUDA segments, and sequential Stage 1 / Stage 2 CPU-GPU offloading for laptops with 4GB-8GB VRAM.",
            value=True,
            invalidate=False,
        ),
        desc.IntParam(
            name="seed",
            label="Random Seed",
            description="Random seed for Stage 1 & Stage 2 diffusion denoising (--seed, range 0 - 2147483647).",
            value=1,
            range=(0, 2147483647, 1),
        ),
        desc.IntParam(
            name="steps",
            label="Denoising Steps",
            description="Number of flow-matching denoising steps for Stage 1 and Stage 2 (--steps, range 10 - 100).",
            value=25,
            range=(10, 100, 1),
        ),
        desc.IntParam(
            name="generateSyntheticViews",
            label="Generate Synthetic Views (360-deg Orbit)",
            description="Number of dense synthetic 360-deg orbital multi-view images to render from the reconstructed 3D model "
                        "(default 24 = renders 24 multi-elevation views at 15, 32, -8 deg into sfmDataDense.abc for Dense MVS; "
                        "0 = disabled / keep only 2-4 sparse views).",
            value=24,
            range=(0, 72, 4),
        ),
        desc.StringParam(
            name="cloudEndpoint",
            label="Cloud Endpoint URL",
            description="Cloud AI inference server URL (falls back to config.json or GENSFM_CLOUD_URL / GENSFM_DEFAULT_ENDPOINT).",
            value="",
            advanced=True,
            invalidate=False,
        ),
        desc.StringParam(
            name="cloudApiKey",
            label="Cloud API Key",
            description="Optional Cloud API key override (falls back to GENSFM_API_KEY / GEMINI_API_KEY in .env so secrets are never stored in .mg files).",
            value="",
            advanced=True,
            invalidate=False,
        ),
        desc.IntParam(
            name="cloudTimeoutSec",
            label="Cloud Timeout (sec)",
            description="Request timeout in seconds for Cloud AI inference.",
            value=300,
            range=(10, 3600, 10),
            advanced=True,
            invalidate=False,
        ),
        desc.StringParam(
            name="pretrainedModel",
            label="Pretrained Model",
            description="Hugging Face repository ID or local path for local_docker / local_venv / local_wsl_gpu (--pretrained).",
            value="dotblue/GenSfm",
            advanced=True,
        ),
        desc.File(
            name="stage1Checkpoint",
            label="Stage 1 Checkpoint",
            description="Optional custom Stage 1 checkpoint path (--stage1 on dotb1ue/GenSfm).",
            value="",
            advanced=True,
        ),
        desc.File(
            name="stage2Checkpoint",
            label="Stage 2 Checkpoint",
            description="Optional custom Stage 2 checkpoint path (--stage2 on dotb1ue/GenSfm).",
            value="",
            advanced=True,
        ),
        desc.BoolParam(
            name="exportAliceVisionSfm",
            label="Export AliceVision SfMData",
            description="Convert predicted cameras.json into AliceVision sfmData.abc and sfmData.sfm for 3D Viewer and downstream nodes.",
            value=True,
            advanced=True,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
            invalidate=False,
        ),
    ]

    outputs = [
        desc.File(
            name="outputSfMData",
            label="SfMData (Alembic - Double-Click to View Cameras)",
            description="Primary sparse SfM output to Meshroom Fast Branch: camera intrinsics, predicted extrinsic poses, and colored sparse point cloud in AliceVision SfMData (.abc) format. Double-click node to view camera frustums in 3D Viewer.",
            semantic="3d",
            value="{nodeCacheFolder}/sfmData.abc",
            group="",
        ),
        desc.File(
            name="outputMeshObj",
            label="Textured Mesh (OBJ - Double-Click to View 3D)",
            description="Textured 3D mesh (.obj + .mtl + mesh_texture.png) in AliceVision coordinate system, ready for 3D Viewer, MeshFiltering_Fast, or Texturing_Fast.",
            semantic="3d",
            value="{nodeCacheFolder}/mesh.obj",
            group="",
        ),
        desc.File(
            name="outputMeshMtl",
            label="Mesh Material (MTL)",
            description="Wavefront material definition linking mesh.obj to mesh_texture.png.",
            value="{nodeCacheFolder}/mesh.mtl",
            group="",
        ),
        desc.File(
            name="outputMeshTexture",
            label="Mesh Texture (PNG)",
            description="360-degree diffuse texture map for mesh.obj.",
            semantic="image",
            value="{nodeCacheFolder}/mesh_texture.png",
            group="",
        ),
        desc.File(
            name="outputMeshGlb",
            label="Textured Mesh (GLB)",
            description="Native GenSfM textured 3D mesh in glTF 2.0 binary (.glb) Y-up (x, z, -y) coordinates for Blender/Unity/Unreal.",
            value="{nodeCacheFolder}/mesh.glb",
            group="",
        ),
        desc.File(
            name="outputGaussianPly",
            label="3D Gaussian Splats (PLY)",
            description="3D Gaussian Splatting representation (.ply), compatible with mrGSplat and 3D point cloud viewers.",
            value="{nodeCacheFolder}/gaussian.ply",
            group="",
        ),
        desc.File(
            name="outputCamerasJson",
            label="GenSfM Cameras JSON",
            description="Raw predicted OpenCV extrinsics (Z-up), normalized intrinsics, and data_to_mesh matrix from GenSfM.",
            value="{nodeCacheFolder}/cameras.json",
            group="",
        ),
        desc.File(
            name="outputMaskedInputs",
            label="Masked RGBA Inputs",
            description="Cropped and background-segmented RGBA object images for inspection in the 2D Viewer.",
            semantic="imageList",
            value="{nodeCacheFolder}/input-rgba-*.png",
            group="",
        ),
        desc.File(
            name="outputReprojections",
            label="Camera Reprojections",
            description="Rendered textured mesh from predicted camera viewpoints to validate pose accuracy in the 2D Viewer.",
            semantic="imageList",
            value="{nodeCacheFolder}/reprojection-*.png",
            group="",
        ),
        desc.File(
            name="outputSfMDataDense",
            label="Dense Synthetic SfMData (24+ Views Alembic)",
            description="Enriched SfMData (.abc) containing the original views plus 24 synthetic 360-deg orbital multi-view renders (synthetic_views/) for PrepareDenseScene_Dense and full MVS photogrammetry.",
            semantic="3d",
            value="{nodeCacheFolder}/sfmDataDense.abc",
            group="",
        ),
        desc.File(
            name="outputSfMDataJson",
            label="SfMData (JSON .sfm)",
            description="Recovered camera poses and intrinsics in AliceVision JSON (.sfm) format for downstream nodes.",
            value="{nodeCacheFolder}/sfmData.sfm",
            group="",
        ),
        desc.File(
            name="outputReprojectionByView",
            label="Reprojection (Active View)",
            description="Rendered mesh reprojection synced with the selected viewpoint in the Image Gallery.",
            semantic="image",
            value="{nodeCacheFolder}/reprojection_<VIEW_ID>.png",
            group="",
        ),
        desc.File(
            name="outputMaskByView",
            label="Foreground Mask (Active View)",
            description="Foreground RGBA mask synced with the selected viewpoint in the Image Gallery.",
            semantic="image",
            value="{nodeCacheFolder}/mask_<VIEW_ID>.png",
            group="",
        ),
    ]

    def processChunk(self, chunk):
        import sys
        from pathlib import Path

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            plugin_root = Path(__file__).resolve().parents[2]
            if str(plugin_root) not in sys.path:
                sys.path.insert(0, str(plugin_root))

            from gensfm_core.env_bootstrap import bootstrap_environment

            bootstrap_environment(plugin_root=plugin_root, logger=chunk.logger)

            from gensfm_core.gensfm_runner import GenSfMRunnerConfig, run_gensfm_pipeline

            input_val = chunk.node.input.value
            if not input_val:
                raise ValueError("GenSfM input attribute is empty. Connect CameraInit.output or provide an image folder.")

            output_dir = Path(chunk.node.outputMeshObj.value).resolve().parent
            output_dir.mkdir(parents=True, exist_ok=True)

            config = GenSfMRunnerConfig(
                input_path=str(input_val),
                output_dir=str(output_dir),
                engine_type=str(chunk.node.engineType.value),
                execution_mode=str(chunk.node.executionBackend.value),
                ai_provider=str(chunk.node.aiProvider.value),
                ai_model_name=str(chunk.node.aiModelName.value),
                masking_mode=str(chunk.node.maskingStrategy.value),
                input_masks_folder=str(chunk.node.inputMasks.value or ""),
                view_selection_mode=str(chunk.node.viewSelectionMode.value),
                selected_view_indices=str(chunk.node.selectedIndices.value),
                low_vram_mode=bool(chunk.node.lowVramMode.value),
                seed=int(chunk.node.seed.value),
                steps=int(chunk.node.steps.value),
                generate_synthetic_views=int(chunk.node.generateSyntheticViews.value),
                cloud_endpoint_url=str(chunk.node.cloudEndpoint.value or ""),
                cloud_api_key=str(chunk.node.cloudApiKey.value or ""),
                cloud_timeout_sec=int(chunk.node.cloudTimeoutSec.value),
                pretrained_model=str(chunk.node.pretrainedModel.value or "dotblue/GenSfm"),
                stage1_checkpoint=str(chunk.node.stage1Checkpoint.value or ""),
                stage2_checkpoint=str(chunk.node.stage2Checkpoint.value or ""),
                export_alicevision_sfm=bool(chunk.node.exportAliceVisionSfm.value),
                verbose_level=str(chunk.node.verboseLevel.value),
            )

            chunk.logger.info(
                f"Starting mrGenSfM: engine={config.engine_type}, backend={config.execution_mode}, "
                f"provider={config.ai_provider}, masking={config.masking_mode}, "
                f"synthetic_views={config.generate_synthetic_views}, low_vram={config.low_vram_mode}"
            )
            result = run_gensfm_pipeline(config=config, logger=chunk.logger)
            chunk.logger.info(
                f"mrGenSfM completed successfully. Mesh: {result.mesh_obj_path}, SfMData: {result.sfm_abc_path}"
            )
        except Exception as exc:
            chunk.logger.error(f"mrGenSfM failed: {exc}")
            raise
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as cleanup_exc:
                chunk.logger.debug(f"CUDA cache cleanup skipped: {cleanup_exc}")
            chunk.logManager.end()
