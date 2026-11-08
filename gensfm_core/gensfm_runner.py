"""
Main Dual-Engine Reconstruction Orchestrator & CLI Dispatcher for mrGenSfM.

Coordinates:
1. Input SfMData / image folder loading & smart 2-4 view selection
2. Foreground RGBA segmentation & crop padding (`masking.py`)
3. Dual Reconstruction Engine execution:
   - Engine 1 (`gensfm_diffusion`): Local WSL2 GPU Bridge (with Low-VRAM offloading),
     Local Docker GPU, Local Venv, or 3D Cloud AI Server
   - Engine 2 (`llm_guided_sfm`): Multimodal LLM Pose Estimation (Gemini / GPT / Claude)
     + Multi-View 3D Visual Hull & Normal-Weighted Texture Baking
4. Standardization of Meshroom 2D & 3D Viewer outputs:
   - `mesh.glb` -> `mesh.obj` + `mesh.mtl` + `mesh_texture.png`
   - `gaussian.ply`
   - `cameras.json` -> `sfmData.sfm` & `sfmData.abc` (via `pyalicevision` / `aliceVision_convertSfMFormat`)
   - `input-rgba-*.png`, `reprojection-*.png`, `mask_<VIEW_ID>.png`, `reprojection_<VIEW_ID>.png`
"""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image

from gensfm_core.camera_converter import (
    convert_cameras_json_to_sfm,
    synthesize_dense_views_and_sfm,
)
from gensfm_core.cloud_client import run_gensfm_cloud_api
from gensfm_core.env_bootstrap import bootstrap_environment, resolve_api_key, resolve_endpoint_url
from gensfm_core.llm_sfm_engine import run_llm_guided_sfm
from gensfm_core.masking import PreparedViewRecord, load_input_views, prepare_masked_views, select_views
from gensfm_core.mesh_converter import _parse_glb_native, convert_glb_to_obj
from gensfm_core.wsl_docker_bridge import (
    run_gensfm_via_docker,
    run_gensfm_via_local_venv,
    run_gensfm_via_wsl,
)


@dataclass
class GenSfMRunnerConfig:
    """Configuration parameters for a single GenSfM node execution."""

    input_path: str
    output_dir: str
    engine_type: str = "gensfm_diffusion"
    execution_mode: str = "cloud_api"
    ai_provider: str = "gemini"
    ai_model_name: str = "gemini-3.8-flash"
    masking_mode: str = "auto_rembg"
    input_masks_folder: str = ""
    view_selection_mode: str = "all_up_to_4"
    selected_view_indices: str = "0,1,2,3"
    low_vram_mode: bool = True
    seed: int = 1
    steps: int = 25
    generate_synthetic_views: int = 24
    cloud_endpoint_url: str = ""
    cloud_api_key: str = ""
    cloud_timeout_sec: int = 300
    pretrained_model: str = "dotblue/GenSfm"
    stage1_checkpoint: str = ""
    stage2_checkpoint: str = ""
    export_alicevision_sfm: bool = True
    verbose_level: str = "info"


@dataclass
class GenSfMRunnerResult:
    """Paths to all generated output artifacts in `{nodeCacheFolder}`."""

    output_dir: Path
    mesh_glb_path: Path
    mesh_obj_path: Path
    gaussian_ply_path: Path
    cameras_json_path: Path
    sfm_json_path: Optional[Path]
    sfm_abc_path: Optional[Path]
    masked_image_paths: List[Path]
    reprojection_paths: List[Path]
    sfm_dense_abc_path: Optional[Path] = None
    synthetic_view_paths: Optional[List[Path]] = None


def _sync_active_view_outputs(
    prepared_views: List[PreparedViewRecord],
    output_dir: Path,
    cameras_json_path: Optional[Path] = None,
) -> List[Path]:
    """
    Ensure `reprojection-{i:02d}.png` and `reprojection_<VIEW_ID>.png` both exist in output_dir,
    and patch `cameras.json`'s `intrinsics_source` if a native backend ran on pre-cropped RGBA inputs
    so that AliceVision SfMData intrinsics match the original CameraInit source images.
    """
    reproj_paths: List[Path] = []
    for idx, pv in enumerate(prepared_views):
        seq_reproj = output_dir / f"reprojection-{idx:02d}.png"
        view_reproj = output_dir / f"reprojection_{pv.view_descriptor.view_id}.png"
        if seq_reproj.is_file() and not view_reproj.is_file():
            shutil.copy2(seq_reproj, view_reproj)
        elif view_reproj.is_file() and not seq_reproj.is_file():
            shutil.copy2(view_reproj, seq_reproj)
        elif not seq_reproj.is_file() and not view_reproj.is_file():
            # Fallback copy from conditioned RGB input if backend did not render reprojections
            shutil.copy2(pv.rgb_cond_path, seq_reproj)
            shutil.copy2(pv.rgb_cond_path, view_reproj)
        reproj_paths.append(seq_reproj)

    if cameras_json_path is not None and cameras_json_path.is_file():
        import json
        import numpy as np
        from gensfm_core.llm_sfm_engine import compute_source_intrinsics

        try:
            cam_data = json.loads(cameras_json_path.read_text(encoding="utf-8"))
            raw_k = cam_data.get("intrinsics", [])
            if isinstance(raw_k, list) and len(raw_k) == len(prepared_views):
                k_arrays = [np.asarray(k, dtype=np.float64) for k in raw_k]
                cam_data["intrinsics_source"] = compute_source_intrinsics(k_arrays, prepared_views)
                cameras_json_path.write_text(json.dumps(cam_data, indent=2) + "\n", encoding="utf-8")
        except Exception as sync_exc:
            logging.getLogger(__name__).debug(
                f"Skipped intrinsics_source synchronization for {cameras_json_path}: {sync_exc}"
            )

    return reproj_paths


def run_gensfm_pipeline(
    config: GenSfMRunnerConfig,
    logger: Optional[logging.Logger] = None,
) -> GenSfMRunnerResult:
    """
    Execute the end-to-end mrGenSfM single-node workflow.
    """
    if logger is None:
        logger = logging.getLogger("mrGenSfM")

    bootstrap_environment(logger=logger)

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load views from CameraInit SfMData (.sfm / .abc) or image directory
    all_views = load_input_views(config.input_path, logger=logger)
    selected_views = select_views(
        views=all_views,
        selection_mode=config.view_selection_mode,
        indices_str=config.selected_view_indices,
        max_views=4,
        logger=logger,
    )
    logger.info(
        f"Selected {len(selected_views)}/{len(all_views)} views: "
        f"{[v.image_path.name for v in selected_views]}"
    )

    # 2. Foreground masking & safe crop preparation
    prepared_views = prepare_masked_views(
        views=selected_views,
        output_dir=output_dir,
        masking_mode=config.masking_mode,
        input_masks_folder=config.input_masks_folder,
        logger=logger,
    )
    rgba_paths = [pv.rgba_path for pv in prepared_views]

    # 3. Dispatch to selected Reconstruction Engine
    api_key = resolve_api_key(config.ai_provider, config.cloud_api_key)
    endpoint_url = resolve_endpoint_url(config.cloud_endpoint_url)
    llm_providers = {"gemini", "openai", "anthropic"}
    has_explicit_cloud_endpoint = bool(config.cloud_endpoint_url.strip())

    use_llm_engine = config.engine_type == "llm_guided_sfm" or (
        config.execution_mode == "cloud_api"
        and config.ai_provider in llm_providers
        and not has_explicit_cloud_endpoint
    )

    if use_llm_engine:
        logger.info(
            f"Running LLM-Guided Sparse SfM + Visual Hull Engine (provider={config.ai_provider}, model={config.ai_model_name})..."
        )
        run_llm_guided_sfm(
            prepared_views=prepared_views,
            output_dir=output_dir,
            ai_provider=config.ai_provider,
            ai_model_name=config.ai_model_name,
            api_key=api_key,
            seed=config.seed,
            logger=logger,
        )
    else:
        # Native GenSfM 3D Diffusion Engine (WSL2 GPU / Docker / Local Venv / 3D Cloud Server)
        try:
            if config.execution_mode == "local_wsl_gpu":
                run_gensfm_via_wsl(
                    image_paths=rgba_paths,
                    output_dir=output_dir,
                    pretrained_model=config.pretrained_model,
                    stage1_checkpoint=config.stage1_checkpoint,
                    stage2_checkpoint=config.stage2_checkpoint,
                    seed=config.seed,
                    steps=config.steps,
                    low_vram_mode=config.low_vram_mode,
                    remove_background=False,
                    logger=logger,
                )
            elif config.execution_mode == "local_docker":
                run_gensfm_via_docker(
                    image_paths=rgba_paths,
                    output_dir=output_dir,
                    pretrained_model=config.pretrained_model,
                    stage1_checkpoint=config.stage1_checkpoint,
                    stage2_checkpoint=config.stage2_checkpoint,
                    seed=config.seed,
                    steps=config.steps,
                    low_vram_mode=config.low_vram_mode,
                    remove_background=False,
                    logger=logger,
                )
            elif config.execution_mode == "local_venv":
                run_gensfm_via_local_venv(
                    image_paths=rgba_paths,
                    output_dir=output_dir,
                    pretrained_model=config.pretrained_model,
                    stage1_checkpoint=config.stage1_checkpoint,
                    stage2_checkpoint=config.stage2_checkpoint,
                    seed=config.seed,
                    steps=config.steps,
                    low_vram_mode=config.low_vram_mode,
                    remove_background=False,
                    logger=logger,
                )
            elif config.execution_mode == "cloud_api":
                run_gensfm_cloud_api(
                    image_paths=rgba_paths,
                    output_dir=output_dir,
                    endpoint_url=endpoint_url,
                    api_key=api_key,
                    provider=config.ai_provider,
                    model_name=config.pretrained_model,
                    seed=config.seed,
                    steps=config.steps,
                    remove_background=(config.masking_mode == "auto_rembg"),
                    timeout_sec=config.cloud_timeout_sec,
                    logger=logger,
                )
            else:
                raise ValueError(f"Unsupported executionMode: {config.execution_mode}")
        except Exception as backend_exc:
            logger.warning(
                f"Native GenSfM backend '{config.execution_mode}' could not complete ({backend_exc}). "
                "Automatically falling back to LLM-Guided Sparse SfM + Visual Hull Engine so pipeline succeeds."
            )
            run_llm_guided_sfm(
                prepared_views=prepared_views,
                output_dir=output_dir,
                ai_provider=config.ai_provider,
                ai_model_name=config.ai_model_name,
                api_key=api_key,
                seed=config.seed,
                logger=logger,
            )

    # 4. Ensure mesh.obj + mesh.mtl + mesh_texture.png exist for Meshroom's 3D Viewer
    mesh_glb_path = output_dir / "mesh.glb"
    mesh_obj_path = output_dir / "mesh.obj"
    gaussian_ply_path = output_dir / "gaussian.ply"
    cameras_json_path = output_dir / "cameras.json"

    if mesh_glb_path.is_file() and not mesh_obj_path.is_file():
        convert_glb_to_obj(mesh_glb_path, mesh_obj_path, logger=logger)

    reproj_paths = _sync_active_view_outputs(prepared_views, output_dir, cameras_json_path=cameras_json_path)

    # 5. Convert cameras.json -> AliceVision sfmData.sfm and sfmData.abc (with RGB landmarks)
    sfm_json_path: Optional[Path] = None
    sfm_abc_path: Optional[Path] = None
    sfm_dense_abc_path: Optional[Path] = None
    synthetic_view_paths: List[Path] = []
    if config.export_alicevision_sfm and cameras_json_path.is_file():
        import numpy as np

        mesh_verts = None
        mesh_cols = None
        try:
            mesh_verts, _, mesh_uvs, mesh_tex = _parse_glb_native(mesh_glb_path)
            if mesh_verts is not None and mesh_uvs is not None and mesh_tex is not None:
                tex_np = np.asarray(mesh_tex.convert("RGB"), dtype=np.float32)
                th, tw, _ = tex_np.shape
                vu_px = np.clip(np.round(mesh_uvs[:, 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
                vv_px = np.clip(np.round((1.0 - mesh_uvs[:, 1]) * (th - 1)).astype(np.int32), 0, th - 1)
                mesh_cols = tex_np[vv_px, vu_px]
        except Exception as glb_exc:
            logger.debug(
                f"Could not sample GLB vertices/colors for SfM landmarks ({glb_exc}); proceeding without landmarks."
            )
            mesh_verts = None
            mesh_cols = None

        sfm_json_path, sfm_abc_path = convert_cameras_json_to_sfm(
            cameras_json_path=cameras_json_path,
            views=selected_views,
            output_sfm_path=output_dir / "sfmData.sfm",
            output_abc_path=output_dir / "sfmData.abc",
            mesh_vertices_for_landmarks=mesh_verts,
            mesh_colors_for_landmarks=mesh_cols,
            logger=logger,
        )

        # 6. Path 2 Integration: Sparse-to-Dense View Synthesis (generateSyntheticViews)
        if config.generate_synthetic_views > 0 and mesh_glb_path.is_file():
            _, sfm_dense_abc_path, synthetic_view_paths = synthesize_dense_views_and_sfm(
                mesh_glb_path=mesh_glb_path,
                cameras_json_path=cameras_json_path,
                original_views=selected_views,
                output_dir=output_dir,
                num_synthetic_views=config.generate_synthetic_views,
                logger=logger,
            )
        elif sfm_abc_path is not None and sfm_abc_path.is_file():
            sfm_dense_abc_path = output_dir / "sfmDataDense.abc"
            shutil.copy2(sfm_abc_path, sfm_dense_abc_path)
            if sfm_json_path is not None and sfm_json_path.is_file():
                shutil.copy2(sfm_json_path, output_dir / "sfmDataDense.sfm")

    return GenSfMRunnerResult(
        output_dir=output_dir,
        mesh_glb_path=mesh_glb_path,
        mesh_obj_path=mesh_obj_path,
        gaussian_ply_path=gaussian_ply_path,
        cameras_json_path=cameras_json_path,
        sfm_json_path=sfm_json_path,
        sfm_abc_path=sfm_abc_path,
        masked_image_paths=rgba_paths,
        reprojection_paths=reproj_paths,
        sfm_dense_abc_path=sfm_dense_abc_path,
        synthetic_view_paths=synthetic_view_paths,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="mrGenSfM CLI Runner for AliceVision Meshroom")
    parser.add_argument("--input", required=True, help="Input SfMData (.sfm/.abc) or image directory")
    parser.add_argument("--output-dir", required=True, help="Output cache directory")
    parser.add_argument("--engine-type", default="gensfm_diffusion", choices=["gensfm_diffusion", "llm_guided_sfm"])
    parser.add_argument(
        "--execution-mode",
        default="cloud_api",
        choices=["cloud_api", "local_wsl_gpu", "local_docker", "local_venv"],
    )
    parser.add_argument("--ai-provider", default="gemini")
    parser.add_argument("--ai-model-name", default="gemini-3.8-flash")
    parser.add_argument("--masking-mode", default="auto_rembg")
    parser.add_argument("--input-masks", default="")
    parser.add_argument("--view-selection-mode", default="all_up_to_4")
    parser.add_argument("--selected-view-indices", default="0,1,2,3")
    parser.add_argument("--low-vram", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--generate-synthetic-views", type=int, default=24)
    parser.add_argument("--cloud-endpoint-url", default="")
    parser.add_argument("--cloud-timeout-sec", type=int, default=300)
    parser.add_argument("--pretrained", default="dotblue/GenSfm")
    parser.add_argument("--stage1", default="")
    parser.add_argument("--stage2", default="")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = GenSfMRunnerConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        engine_type=args.engine_type,
        execution_mode=args.execution_mode,
        ai_provider=args.ai_provider,
        ai_model_name=args.ai_model_name,
        masking_mode=args.masking_mode,
        input_masks_folder=args.input_masks,
        view_selection_mode=args.view_selection_mode,
        selected_view_indices=args.selected_view_indices,
        low_vram_mode=args.low_vram,
        seed=args.seed,
        steps=args.steps,
        generate_synthetic_views=args.generate_synthetic_views,
        cloud_endpoint_url=args.cloud_endpoint_url,
        cloud_timeout_sec=args.cloud_timeout_sec,
        pretrained_model=args.pretrained,
        stage1_checkpoint=args.stage1,
        stage2_checkpoint=args.stage2,
    )
    run_gensfm_pipeline(cfg)


if __name__ == "__main__":
    main()
