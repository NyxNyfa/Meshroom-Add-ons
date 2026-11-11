__version__ = "1.0"

import os
import sys
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class AiRemoveBackground(desc.Node):
    """
    AI Foreground Segmentation & Background Removal Node (`AiRemoveBackground`)
    for AliceVision Meshroom 2025.1.0+.

    Removes complex backgrounds (photo backdrops, studio tables, shadows) using
    deep learning models (U2-Net, BiRefNet, IS-Net, or RMBG-2.0) via native Python or WSL2 GPU.
    Generates clean binary alpha masks (`mask_<VIEW_ID>.png`) and transparent RGBA images
    (`rgba_<VIEW_ID>.png`) ready to feed `GenSfM.inputMasks` or standard Meshroom nodes.
    """

    gpu = desc.Level.NORMAL
    size = desc.DynamicNodeSize("input")
    category = "Image Processing / AI"
    documentation = """
AI Foreground Segmentation & Background Removal Node (`AiRemoveBackground`).

### How to Use in Meshroom:
1. Connect **`CameraInit.output`** to **`AiRemoveBackground.input`**.
2. Connect **`AiRemoveBackground.outputFolder`** to **`GenSfM.inputMasks`**.
3. Select the desired AI model:
   - **`u2net`** (*Recommended*): Apache-2.0 open-source model with excellent balance of speed and hair/edge separation.
   - **`birefnet-general`**: High-resolution bilateral reference network for razor-sharp boundaries.
   - **`isnet-general-use`**: High-detail segmentation for intricate shapes and thin structures.
   - **`silueta`**: Ultra-lightweight model for constrained systems.
   - **`briaai-rmbg-2.0`**: State-of-the-art segmentation (free for non-commercial research; requires license for commercial use).
4. Click on `AiRemoveBackground` in the Graph Editor to inspect the cutouts in the **2D Image Viewer**.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file (.sfm / .abc) from CameraInit or a directory containing input images.",
            value="",
        ),
        desc.ChoiceParam(
            name="modelType",
            label="AI Model",
            description="Deep learning segmentation model architecture to use.",
            value="u2net",
            values=["u2net", "birefnet-general", "isnet-general-use", "silueta", "briaai-rmbg-2.0"],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="executionBackend",
            label="Execution Backend",
            description="Execution environment (auto detects local Python or WSL2 GPU conda environment).",
            value="auto",
            values=["auto", "local_wsl_gpu", "local_venv"],
            exclusive=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="saveRgba",
            label="Save Cutout RGBA",
            description="Save transparent RGBA PNG images alongside grayscale alpha masks.",
            value=True,
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
            name="outputFolder",
            label="Output Folder",
            description="Folder containing all generated alpha masks and transparent cutouts for downstream nodes.",
            semantic="imageFolder",
            value="{nodeCacheFolder}",
            group="",
        ),
        desc.File(
            name="outputMasks",
            label="Foreground Mask",
            description="Grayscale alpha mask synced with the selected image in the 2D Image Gallery.",
            semantic="image",
            value="{nodeCacheFolder}/mask_<VIEW_ID>.png",
            group="",
        ),
        desc.File(
            name="outputRgba",
            label="Cutout RGBA",
            description="Cutout transparent RGBA image synced with the selected image in the 2D Image Gallery.",
            semantic="image",
            value="{nodeCacheFolder}/rgba_<VIEW_ID>.png",
            group="",
        ),
    ]

    def processChunk(self, chunk):
        from pathlib import Path

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            plugin_root = Path(__file__).resolve().parents[2]
            if str(plugin_root) not in sys.path:
                sys.path.insert(0, str(plugin_root))

            from gensfm_core.env_bootstrap import bootstrap_environment

            bootstrap_environment(plugin_root=plugin_root, logger=chunk.logger)

            from gensfm_core.masking import _run_ai_masking_wsl, load_input_views

            input_val = chunk.node.input.value
            if not input_val:
                raise ValueError("Input is empty. Connect CameraInit.output or provide an image folder.")

            output_dir = Path(chunk.node.outputFolder.value).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            views = load_input_views(input_val, logger=chunk.logger)
            chunk.logger.info(f"Loaded {len(views)} views for AI background removal.")

            model_type = str(chunk.node.modelType.value)
            # Map briaai-rmbg-2.0 to birefnet-general if rembg engine is used
            model_key = "birefnet-general" if model_type == "briaai-rmbg-2.0" else model_type

            # Check if local rembg is available first
            has_local = False
            try:
                import rembg

                has_local = True
            except Exception:
                has_local = False

            raw_paths = [v.image_path for v in views]
            success = False

            if has_local and chunk.node.executionBackend.value in ("auto", "local_venv"):
                chunk.logger.info(f"Running AI background removal locally using rembg ({model_key})...")
                from PIL import Image

                session = rembg.new_session(model_key)
                for v in views:
                    with Image.open(v.image_path) as im:
                        out = rembg.remove(im, session=session).convert("RGBA")
                    alpha = out.getchannel("A")
                    alpha.save(output_dir / f"mask_{v.image_path.stem}.png")
                    alpha.save(output_dir / f"mask_{v.view_id}.png")
                    out.save(output_dir / f"rgba_{v.image_path.stem}.png")
                    out.save(output_dir / f"rgba_{v.view_id}.png")
                success = True
            else:
                chunk.logger.info(f"Running AI background removal via WSL2 GPU using rembg ({model_key})...")
                success = _run_ai_masking_wsl(
                    raw_paths,
                    output_dir,
                    model_name=model_key,
                    logger=chunk.logger,
                )
                if success:
                    # Create view_id alias copies for Meshroom 2D Image Viewer active viewpoint sync
                    for v in views:
                        stem_mask = output_dir / f"mask_{v.image_path.stem}.png"
                        vid_mask = output_dir / f"mask_{v.view_id}.png"
                        if stem_mask.is_file() and not vid_mask.is_file():
                            vid_mask.write_bytes(stem_mask.read_bytes())

                        stem_rgba = output_dir / f"rgba_{v.image_path.stem}.png"
                        vid_rgba = output_dir / f"rgba_{v.view_id}.png"
                        if stem_rgba.is_file() and not vid_rgba.is_file():
                            vid_rgba.write_bytes(stem_rgba.read_bytes())

            if not success:
                raise RuntimeError(
                    f"AI background removal failed for {len(views)} images. "
                    "Ensure WSL2 Ubuntu with the 'gensfm' conda environment is installed, or install 'rembg' locally."
                )

            chunk.logger.info(f"AI background removal finished successfully: {output_dir}")

        except Exception as exc:
            chunk.logger.error(f"AiRemoveBackground failed: {exc}", exc_info=True)
            raise
        finally:
            chunk.logManager.end()

