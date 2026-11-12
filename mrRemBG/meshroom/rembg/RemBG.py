__version__ = "1.0"

import os
from pathlib import Path
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class RemBG(desc.Node):
    """
    AI-Powered Automatic Background Removal Node (`RemBG`) for AliceVision Meshroom 2025.1.0+.

    Removes background from single images or full SfMData camera sets using SOTA deep learning matting
    (U2-Net, U2-Net-p, ISNet, BiRefNet). Produces clean transparent RGBA PNGs and grayscale masks
    designed to feed directly into:
    - `mrAIGen3D` (Single-image 3D generation: TripoSR & TripoSplat)
    - `mrGenSfM` (`inputMasks` or `input` SfMData for sparse reconstruction)
    - Standard AliceVision photogrammetry pipelines.
    """

    gpu = desc.Level.NORMAL
    size = desc.DynamicNodeSize("input")
    category = "AI Generation / Segmentation"
    documentation = """
AI-Powered Automatic Background Removal Node (`RemBG`) for AliceVision Meshroom.

### Overview
Extracts high-precision foreground silhouettes and removes complex backgrounds from photos
before feeding them to 3D reconstruction algorithms:
- **Direct Downstream to mrAIGen3D**: Feed `outputSfm` or `outputFolder` directly into `AI3DGenerator`.
- **Direct Downstream to mrGenSfM**: Connect `outputMasks` directly to `GenSfM.inputMasks`
  (with `maskingStrategy = 'external_masks'`), or connect `outputSfm` directly to `GenSfM.input`
  (with `maskingStrategy = 'existing_alpha'`).

### Multi-View & Single-View Support
Accepts either:
1. `CameraInit.output` (`cameraInit.sfm` containing 1 or more images)
2. A folder containing raw images (`.jpg`, `.png`, `.webp`, `.bmp`)
3. A single image file (.png, .jpg, .webp, .bmp) directly via `input`.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData (.sfm / .json) from CameraInit, image folder, or direct single image file.",
            value="",
        ),
        desc.ChoiceParam(
            name="model",
            label="Segmentation Model",
            description="Underlying background removal neural network:\n"
                        " - u2net: Standard high quality general purpose matting.\n"
                        " - u2netp: Ultra lightweight and fast version of U2-Net.\n"
                        " - isnet-general-use: Highly accurate boundary segmentation.\n"
                        " - birefnet-general: SOTA bilateral reference matting.",
            value="u2net",
            values=["u2net", "u2netp", "isnet-general-use", "birefnet-general"],
            exclusive=True,
        ),
        desc.BoolParam(
            name="alphaMatting",
            label="Alpha Matting",
            description="Refines edge transparency using morphological trimaps and guided matting.\n"
                        "Recommended for fine details like fur, hair, and soft edges.",
            value=False,
        ),
        desc.IntParam(
            name="alphaMattingForegroundThreshold",
            label="Foreground Threshold",
            description="Trimap foreground threshold value (0-255).",
            value=240,
            range=(0, 255, 1),
        ),
        desc.IntParam(
            name="alphaMattingBackgroundThreshold",
            label="Background Threshold",
            description="Trimap background threshold value (0-255).",
            value=10,
            range=(0, 255, 1),
        ),
        desc.IntParam(
            name="alphaMattingErodeSize",
            label="Erode Size",
            description="Erosion kernel size for trimap generation.",
            value=10,
            range=(1, 50, 1),
        ),
        desc.BoolParam(
            name="padToSquare",
            label="Pad To Square",
            description="Pads the output transparent image to a 1:1 aspect ratio with transparent borders.\n"
                        "Recommended for 3D generative models like TripoSR, TripoSplat, and GenSfM.",
            value=False,
        ),
        desc.ChoiceParam(
            name="executionBackend",
            label="Execution Backend",
            description="Environment used to execute background removal:\n"
                        " - local_wsl_gpu: Use active CUDA-accelerated WSL2 environment.\n"
                        " - local_windows: Use Windows Python environment if rembg is installed.\n"
                        " - local_cpu: Force CPU computation.",
            value="local_wsl_gpu",
            values=["local_wsl_gpu", "local_windows", "local_cpu"],
            exclusive=True,
        ),
        desc.StringParam(
            name="wslDistro",
            label="WSL Distro",
            description="Target WSL2 distribution name.",
            value="Ubuntu",
        ),
        desc.StringParam(
            name="wslCondaEnv",
            label="WSL Conda Env",
            description="Target conda environment inside WSL2 that contains rembg and PyTorch.",
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
            name="outputSfm",
            label="Output SfMData",
            description="Updated SfMData JSON with camera viewpoints referencing the segmented transparent PNG images.",
            value="{nodeCacheFolder}/sfmData.sfm",
        ),
        desc.File(
            name="outputFolder",
            label="RGBA Images Folder",
            description="Directory containing all segmented RGBA transparent PNG images.",
            value="{nodeCacheFolder}/rgba_images",
        ),
        desc.File(
            name="outputMasks",
            label="Masks Folder",
            description="Directory containing grayscale alpha masks (white = object, black = bg). Connect directly to GenSfM.inputMasks.",
            value="{nodeCacheFolder}/masks",
        ),
        desc.File(
            name="outputPreview",
            label="Preview Image",
            description="Segmented transparent preview synced with 2D Viewer.",
            semantic="image",
            value="{nodeCacheFolder}/preview_<VIEW_ID>.png",
            group="",
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

        from rembg_core.sfm_io import SfmDataIO
        from rembg_core.wsl_bridge import run_rembg_in_wsl, is_wsl_available

        input_val = str(chunk.node.input.value or "").strip()
        if not input_val:
            raise ValueError("mrRemBG requires an 'input' (SfMData from CameraInit, image folder, or single image file).")

        io_handler = SfmDataIO(logger=logger)
        views, orig_dict = io_handler.load_views(input_val)
        if not views:
            raise ValueError("No views found to process in mrRemBG.")

        output_sfm = Path(chunk.node.outputSfm.value).resolve()
        output_rgba_dir = Path(chunk.node.outputFolder.value).resolve()
        output_masks_dir = Path(chunk.node.outputMasks.value).resolve()
        cache_dir = output_sfm.parent

        output_rgba_dir.mkdir(parents=True, exist_ok=True)
        output_masks_dir.mkdir(parents=True, exist_ok=True)

        backend = str(chunk.node.executionBackend.value)
        model_name = str(chunk.node.model.value)
        alpha_matting = bool(chunk.node.alphaMatting.value)
        af_thresh = int(chunk.node.alphaMattingForegroundThreshold.value)
        ab_thresh = int(chunk.node.alphaMattingBackgroundThreshold.value)
        erode_size = int(chunk.node.alphaMattingErodeSize.value)
        pad_to_square = bool(chunk.node.padToSquare.value)

        # Decide whether to execute via WSL or local python
        use_wsl = (backend == "local_wsl_gpu") or (backend == "local_cpu" and is_wsl_available())

        if use_wsl:
            logger.info("Delegating background removal to WSL2 environment...")
            input_paths = [v.image_path for v in views]
            run_rembg_in_wsl(
                input_paths=input_paths,
                output_rgba_dir=output_rgba_dir,
                output_mask_dir=output_masks_dir,
                output_sfm_path=output_sfm,
                original_sfm_path=Path(input_val).resolve() if Path(input_val).is_file() else None,
                model_name=model_name,
                alpha_matting=alpha_matting,
                af_thresh=af_thresh,
                ab_thresh=ab_thresh,
                erode_size=erode_size,
                pad_to_square=pad_to_square,
                distro=str(chunk.node.wslDistro.value),
                conda_env=str(chunk.node.wslCondaEnv.value),
                logger=logger,
            )
        else:
            logger.info("Executing background removal in host Python environment...")
            from rembg_core.engine import RemBGEngine
            engine = RemBGEngine(
                model_name=model_name,
                alpha_matting=alpha_matting,
                af_thresh=af_thresh,
                ab_thresh=ab_thresh,
                erode_size=erode_size,
                logger=logger,
            )
            processed_records = []
            for idx, item in enumerate(views):
                base_name = item.image_path.stem
                rgba_out = output_rgba_dir / f"{base_name}_rgba.png"
                mask_out = output_masks_dir / f"{base_name}_mask.png"
                logger.info("[%d/%d] Processing %s...", idx + 1, len(views), item.image_path.name)
                engine.process_image(
                    input_image_path=item.image_path,
                    output_rgba_path=rgba_out,
                    output_mask_path=mask_out,
                    pad_to_square=pad_to_square,
                )
                processed_records.append((item, rgba_out, mask_out))

            io_handler.write_output_sfm(
                output_sfm_path=output_sfm,
                original_sfm_dict=orig_dict,
                processed_views=processed_records,
            )

        # Generate preview files for Meshroom 2D Image Viewer
        logger.info("Generating 2D Image Viewer preview files...")
        for idx, item in enumerate(views):
            base_name = item.image_path.stem
            rgba_out = output_rgba_dir / f"{base_name}_rgba.png"
            if rgba_out.is_file():
                # Provide standard preview_<VIEW_ID>.png
                preview_dst = cache_dir / f"preview_{item.view_id}.png"
                try:
                    shutil.copyfile(rgba_out, preview_dst)
                except Exception:
                    pass

        logger.info("mrRemBG chunk processing successfully completed.")

