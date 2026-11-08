"""
Coordinate & Geometry Bridge (`scripts/sfm_bridge.py`) for mrGenSfM.

Converts GenSfM's `cameras.json` (OpenCV Z-up world-to-camera extrinsics, normalized intrinsics,
and `data_to_mesh` basis matrix) and `mesh.glb` into AliceVision `sfmData.abc`, `sfmData.sfm`,
`sfmDataDense.abc`, and `mesh.obj` + `mesh.mtl` + `mesh_texture.png`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gensfm_core.camera_converter import (
    DEFAULT_DATA_TO_MESH,
    convert_cameras_json_to_sfm,
    gensfm_extrinsic_to_alicevision_pose,
    normalized_intrinsic_to_alicevision,
    synthesize_dense_views_and_sfm,
)
from gensfm_core.masking import load_input_views, select_views
from gensfm_core.mesh_converter import (
    _parse_glb_native,
    convert_glb_to_obj,
    write_gaussian_ply,
    write_textured_glb,
    write_textured_obj,
)

__all__ = [
    "DEFAULT_DATA_TO_MESH",
    "convert_cameras_json_to_sfm",
    "gensfm_extrinsic_to_alicevision_pose",
    "normalized_intrinsic_to_alicevision",
    "synthesize_dense_views_and_sfm",
    "convert_glb_to_obj",
    "write_textured_glb",
    "write_textured_obj",
    "write_gaussian_ply",
]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert GenSfM cameras.json & mesh.glb into AliceVision sfmData.abc and mesh.obj"
    )
    parser.add_argument("--cameras-json", required=True, type=Path, help="Path to cameras.json")
    parser.add_argument("--input-views", required=True, type=Path, help="Input CameraInit .sfm/.abc or image folder")
    parser.add_argument("--mesh-glb", type=Path, default=None, help="Optional path to mesh.glb")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--synthetic-views", type=int, default=0, help="Number of synthetic dense views to render")
    args = parser.parse_args(argv)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    views = select_views(load_input_views(args.input_views), selection_mode="all_up_to_4")
    mesh_verts = None
    mesh_cols = None
    if args.mesh_glb and args.mesh_glb.is_file():
        import numpy as np

        convert_glb_to_obj(args.mesh_glb, out_dir / "mesh.obj")
        mesh_verts, _, mesh_uvs, mesh_tex = _parse_glb_native(args.mesh_glb)
        if mesh_verts is not None and mesh_uvs is not None and mesh_tex is not None:
            tex_np = np.asarray(mesh_tex.convert("RGB"), dtype=np.float32)
            th, tw, _ = tex_np.shape
            vu_px = np.clip(np.round(mesh_uvs[:, 0] * (tw - 1)).astype(np.int32), 0, tw - 1)
            vv_px = np.clip(np.round((1.0 - mesh_uvs[:, 1]) * (th - 1)).astype(np.int32), 0, th - 1)
            mesh_cols = tex_np[vv_px, vu_px]

    convert_cameras_json_to_sfm(
        cameras_json_path=args.cameras_json,
        views=views,
        output_sfm_path=out_dir / "sfmData.sfm",
        output_abc_path=out_dir / "sfmData.abc",
        mesh_vertices_for_landmarks=mesh_verts,
        mesh_colors_for_landmarks=mesh_cols,
    )
    if args.synthetic_views > 0 and args.mesh_glb and args.mesh_glb.is_file():
        synthesize_dense_views_and_sfm(
            mesh_glb_path=args.mesh_glb,
            cameras_json_path=args.cameras_json,
            original_views=views,
            output_dir=out_dir,
            num_synthetic_views=args.synthetic_views,
        )


if __name__ == "__main__":
    main()

