"""
Comprehensive Unit & End-to-End Integration Test Suite for mrGenSfM.

Tests:
1. Environment bootstrap, .env parsing, API key & endpoint resolution
2. Windows <-> WSL2 path conversion & edge cases
3. View loading (.sfm & directory), view selection (manual, uniform_spread, ai_smart_select),
   and foreground RGBA masking (existing_alpha, auto_grabcut, external_masks)
4. Binary glTF 2.0 (.glb) <-> Wavefront (.obj + .mtl + texture.png) roundtrip & 3DGS (.ply) export
5. OpenCV Z-up cameras.json -> AliceVision Y-up sfmData.sfm & sfmData.abc conversion
6. Full end-to-end pipeline execution (`run_gensfm_pipeline`) on real multi-view images
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from gensfm_core.camera_converter import (
    DEFAULT_DATA_TO_MESH,
    convert_cameras_json_to_sfm,
    gensfm_extrinsic_to_alicevision_pose,
    normalized_intrinsic_to_alicevision,
)
from gensfm_core.env_bootstrap import (
    bootstrap_environment,
    load_dotenv_file,
    resolve_api_key,
    resolve_endpoint_url,
)
from gensfm_core.gensfm_runner import GenSfMRunnerConfig, run_gensfm_pipeline
from gensfm_core.llm_sfm_engine import normalize_gemini_model_name
from gensfm_core.masking import (
    load_input_views,
    prepare_masked_views,
    select_views,
)
from gensfm_core.mesh_converter import (
    convert_glb_to_obj,
    write_gaussian_ply,
    write_textured_glb,
)
from gensfm_core.wsl_docker_bridge import windows_to_wsl_path, wsl_to_windows_path


class TestMrGenSfMPipeline(unittest.TestCase):
    def setUp(self) -> None:
        bootstrap_environment(plugin_root=PLUGIN_ROOT)
        self.tmp_dir_obj = tempfile.TemporaryDirectory(prefix="test_mrgensfm_")
        self.tmp_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def _create_synthetic_views(self, folder: Path, count: int = 5) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            arr = np.zeros((128, 128, 4), dtype=np.uint8)
            # Draw a colored foreground circle shifting slightly across views
            yy, xx = np.ogrid[:128, :128]
            cx = 64 + (i - count // 2) * 3
            cy = 64
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (34 ** 2)
            arr[mask, 0] = (50 + i * 40) % 255
            arr[mask, 1] = (120 + i * 25) % 255
            arr[mask, 2] = (200 - i * 30) % 255
            arr[mask, 3] = 255
            Image.fromarray(arr, mode="RGBA").save(folder / f"{i:02d}.png")
        return folder

    def test_env_bootstrap_and_dotenv(self) -> None:
        env_file = self.tmp_dir / ".env"
        env_file.write_text(
            "# Comment line\nTEST_GENSFM_CUSTOM_VAR='hello_world'\nGENSFM_DEFAULT_ENDPOINT=http://localhost:9123/\n",
            encoding="utf-8",
        )
        parsed = load_dotenv_file(env_file)
        self.assertEqual(parsed.get("TEST_GENSFM_CUSTOM_VAR"), "hello_world")
        self.assertEqual(resolve_api_key("gemini", override_key="  my_override_key "), "my_override_key")
        self.assertEqual(resolve_endpoint_url("http://127.0.0.1:8080/"), "http://127.0.0.1:8080")
        self.assertEqual(normalize_gemini_model_name("gemini-2.5-flash"), "gemini-3.8-flash")
        self.assertEqual(normalize_gemini_model_name("gemini-2.0-flash"), "gemini-3.8-flash")
        self.assertEqual(normalize_gemini_model_name("gemini-3.8-flash"), "gemini-3.8-flash")

    def test_wsl_path_conversions(self) -> None:
        self.assertEqual(
            windows_to_wsl_path(r"D:\Meshroom\mrGenSfM\output"),
            "/mnt/d/Meshroom/mrGenSfM/output",
        )
        self.assertEqual(windows_to_wsl_path("/mnt/c/Users/test"), "/mnt/c/Users/test")
        self.assertEqual(
            wsl_to_windows_path("/mnt/d/Meshroom/GenSfm"),
            r"D:\Meshroom\GenSfm",
        )
        with self.assertRaises(ValueError):
            windows_to_wsl_path("   ")

    def test_view_loading_selection_and_masking_edge_cases(self) -> None:
        empty_dir = self.tmp_dir / "empty"
        empty_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            load_input_views(empty_dir)
        self.assertIn("GenSfM requires at least 2 views of the object", str(ctx.exception))

        img_dir = self._create_synthetic_views(self.tmp_dir / "synth_views", count=6)
        views = load_input_views(img_dir)
        self.assertEqual(len(views), 6)

        sel_up_to_4 = select_views(views, "all_up_to_4", "")
        self.assertEqual([v.frame_id for v in sel_up_to_4], [0, 1, 2, 3])

        sel_diverse = select_views(views, "auto_diverse", "")
        self.assertEqual(len(sel_diverse), 4)

        sel_manual = select_views(views, "manual_indices", "0,2,4,5")
        self.assertEqual([v.frame_id for v in sel_manual], [0, 2, 4, 5])

        sel_uniform = select_views(views, "uniform_spread", "")
        self.assertEqual(len(sel_uniform), 4)
        self.assertEqual(sel_uniform[0].frame_id, 0)
        self.assertEqual(sel_uniform[-1].frame_id, 5)

        sel_smart = select_views(views, "ai_smart_select", "")
        self.assertEqual(len(sel_smart), 4)

        out_masks = self.tmp_dir / "masked_out"
        prepared = prepare_masked_views(
            sel_uniform,
            output_dir=out_masks,
            masking_mode="auto_grabcut",
            target_resolution=128,
        )
        self.assertEqual(len(prepared), 4)
        for idx, pv in enumerate(prepared):
            self.assertTrue(pv.rgba_path.is_file())
            self.assertTrue(pv.rgb_cond_path.is_file())
            self.assertTrue(pv.gallery_mask_path.is_file())
            self.assertTrue((out_masks / "prepared_inputs" / f"{idx:02d}.png").is_file())

        # Test AI rembg masking fallback / WSL bridge
        out_ai = self.tmp_dir / "masked_ai"
        prepared_ai = prepare_masked_views(
            sel_uniform[:2],
            output_dir=out_ai,
            masking_mode="auto_rembg",
            target_resolution=128,
        )
        self.assertEqual(len(prepared_ai), 2)
        for pv in prepared_ai:
            self.assertTrue(pv.rgba_path.is_file())
            self.assertTrue(pv.rgb_cond_path.is_file())

    def test_glb_obj_and_gaussian_ply_roundtrip(self) -> None:
        verts = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        tex = Image.new("RGB", (64, 64), (180, 90, 45))

        glb_path = write_textured_glb(self.tmp_dir / "test.glb", verts, faces, uvs, tex)
        self.assertTrue(glb_path.is_file())
        self.assertGreater(glb_path.stat().st_size, 100)

        obj_path, mtl_path, tex_path = convert_glb_to_obj(glb_path, self.tmp_dir / "test.obj")
        self.assertTrue(obj_path.is_file())
        self.assertTrue(mtl_path.is_file())
        self.assertTrue(tex_path.is_file())
        obj_text = obj_path.read_text(encoding="utf-8")
        self.assertIn("mtllib test.mtl", obj_text)
        self.assertIn("vt ", obj_text)
        self.assertIn("f 1/1/1", obj_text)

        ply_path = write_gaussian_ply(
            self.tmp_dir / "gaussian.ply",
            positions=verts,
            colors_rgb_01=np.ones_like(verts) * 0.5,
        )
        self.assertTrue(ply_path.is_file())
        raw_ply = ply_path.read_bytes()
        self.assertTrue(raw_ply.startswith(b"ply\nformat binary_little_endian 1.0\n"))
        self.assertIn(b"property float f_dc_0", raw_ply)
        self.assertIn(b"property uchar red", raw_ply)

    def test_camera_converter_and_alicevision_sfm(self) -> None:
        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = [0.0, 0.0, 2.0]
        r_mesh, c_mesh = gensfm_extrinsic_to_alicevision_pose(ext.tolist(), DEFAULT_DATA_TO_MESH)
        self.assertAlmostEqual(float(np.linalg.det(r_mesh)), 1.0, places=6)
        self.assertEqual(c_mesh.shape, (3,))

        bad_ext = ext.copy()
        bad_ext[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            gensfm_extrinsic_to_alicevision_pose(bad_ext.tolist(), DEFAULT_DATA_TO_MESH)

        intr_dict = normalized_intrinsic_to_alicevision(
            [[1.2, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]],
            width=1024,
            height=1024,
        )
        self.assertAlmostEqual(intr_dict["focalLength"], 1.2 * 36.0, places=4)
        self.assertAlmostEqual(intr_dict["principalPoint"][0], 0.0, places=4)

        with self.assertRaises(ValueError):
            normalized_intrinsic_to_alicevision(
                [[float("inf"), 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]],
                width=1024,
                height=1024,
            )

    def test_boundary_foreground_padding_and_gemini_schema_mock(self) -> None:
        from unittest.mock import MagicMock, patch
        from gensfm_core.llm_sfm_engine import _query_gemini_poses

        # 1. Create 2 views where the foreground object touches the very edge of the image (0..127)
        edge_dir = self.tmp_dir / "edge_views"
        edge_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            arr = np.zeros((100, 100, 4), dtype=np.uint8)
            arr[0:100, 0:100, :3] = 180
            arr[0:100, 0:100, 3] = 255
            Image.fromarray(arr, mode="RGBA").save(edge_dir / f"{i:02d}.png")

        views = load_input_views(edge_dir)
        prepared = prepare_masked_views(
            views,
            output_dir=self.tmp_dir / "edge_out",
            masking_mode="existing_alpha",
            target_resolution=128,
        )
        self.assertEqual(len(prepared), 2)
        for pv in prepared:
            # Verify source_size matches original 100x100 image, not padded canvas
            self.assertEqual(pv.source_size, (100, 100))
            # Verify outer border of saved input-rgba-*.png is transparent so native GenSfM crop_size_ratio=1.2 never fails
            self.assertEqual(int(pv.rgba_array[0, :, 3].max()), 0)
            self.assertEqual(int(pv.rgba_array[-1, :, 3].max()), 0)
            self.assertEqual(int(pv.rgba_array[:, 0, 3].max()), 0)
            self.assertEqual(int(pv.rgba_array[:, -1, 3].max()), 0)

        # 2. Verify Gemini Interactions API REST payload uses current response_format schema
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "cameras": [
                                        {
                                            "index": 0,
                                            "azimuth_deg": 0.0,
                                            "elevation_deg": 15.0,
                                            "roll_deg": 0.0,
                                            "distance": 2.0,
                                            "fov_deg": 40.0,
                                        },
                                        {
                                            "index": 1,
                                            "azimuth_deg": 90.0,
                                            "elevation_deg": 20.0,
                                            "roll_deg": 0.0,
                                            "distance": 2.1,
                                            "fov_deg": 40.0,
                                        },
                                    ]
                                }
                            ),
                        }
                    ],
                }
            ]
        }
        with patch("requests.post", return_value=mock_resp) as mocked_post:
            with patch.dict("sys.modules", {"google": None, "google.genai": None}):
                cams = _query_gemini_poses(
                    prepared,
                    api_key="test_dummy_key",
                    model_name="gemini-2.5-flash",
                    seed=7,
                )
                self.assertEqual(len(cams), 2)
                self.assertAlmostEqual(cams[1].azimuth_deg, 90.0)
                called_json = mocked_post.call_args.kwargs["json"]
                self.assertEqual(called_json["model"], "models/gemini-3.8-flash")
                self.assertEqual(called_json["response_format"]["type"], "text")
                self.assertEqual(called_json["response_format"]["mime_type"], "application/json")
                self.assertIn("schema", called_json["response_format"])

    def test_end_to_end_pipeline_on_wagon_or_synthetic(self) -> None:
        wagon_dir = Path("D:/Meshroom/GenSfm/assets/example/wagon")
        if wagon_dir.is_dir():
            input_source = wagon_dir
        else:
            input_source = self._create_synthetic_views(self.tmp_dir / "fallback_wagon", count=4)

        out_dir = self.tmp_dir / "pipeline_out"
        cfg = GenSfMRunnerConfig(
            input_path=str(input_source),
            output_dir=str(out_dir),
            engine_type="llm_guided_sfm",
            execution_mode="cloud_api",
            ai_provider="gemini",
            masking_mode="existing_alpha",
            view_selection_mode="all_up_to_4",
            seed=42,
            steps=10,
            generate_synthetic_views=8,
            export_alicevision_sfm=True,
        )
        result = run_gensfm_pipeline(cfg)

        self.assertTrue(result.mesh_glb_path.is_file())
        self.assertTrue(result.mesh_obj_path.is_file())
        self.assertTrue((out_dir / "mesh.mtl").is_file())
        self.assertTrue((out_dir / "mesh_texture.png").is_file())
        self.assertTrue(result.gaussian_ply_path.is_file())
        self.assertTrue(result.cameras_json_path.is_file())
        self.assertIsNotNone(result.sfm_json_path)
        self.assertTrue(result.sfm_json_path.is_file())
        self.assertIsNotNone(result.sfm_abc_path)
        self.assertTrue(result.sfm_abc_path.is_file())
        self.assertIsNotNone(result.sfm_dense_abc_path)
        self.assertTrue(result.sfm_dense_abc_path.is_file())
        self.assertIsNotNone(result.synthetic_view_paths)
        self.assertEqual(len(result.synthetic_view_paths), 8)
        self.assertEqual(len(result.masked_image_paths), 4)
        self.assertEqual(len(result.reprojection_paths), 4)

        sfm_data = json.loads(result.sfm_json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(sfm_data["views"]), 4)
        self.assertEqual(len(sfm_data["poses"]), 4)
        self.assertGreaterEqual(len(sfm_data["intrinsics"]), 1)
        self.assertGreater(len(sfm_data.get("structure", [])), 2000)

        # Verify 100% outward-pointing triangle normals on reconstructed mesh.glb
        from gensfm_core.mesh_converter import _parse_glb_native

        verts_glb, faces_glb, _, _ = _parse_glb_native(result.mesh_glb_path)
        self.assertEqual(len(sfm_data["structure"]), len(verts_glb))
        v0, v1, v2 = verts_glb[faces_glb[:, 0]], verts_glb[faces_glb[:, 1]], verts_glb[faces_glb[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_centers = (v0 + v1 + v2) / 3.0
        outward_dots = np.sum(face_normals * face_centers, axis=1)
        self.assertEqual(int(np.count_nonzero(outward_dots <= 0.0)), 0)

        # Verify AliceVision .sfm rotation is stored as camera-to-world (r_mesh.T) so R_w2c @ (0 - C) == t_mesh
        cam_json = json.loads(result.cameras_json_path.read_text(encoding="utf-8"))
        for idx, pose_entry in enumerate(sfm_data["poses"]):
            r_c2w = np.array([float(x) for x in pose_entry["pose"]["transform"]["rotation"]]).reshape(3, 3)
            c_world = np.array([float(x) for x in pose_entry["pose"]["transform"]["center"]])
            r_w2c = r_c2w.T
            t_recovered = -(r_w2c @ c_world)
            t_expected = np.asarray(cam_json["extrinsics"][idx], dtype=np.float64)[:3, 3]
            np.testing.assert_allclose(t_recovered, t_expected, atol=1e-5)

        # Verify multi-view landmark observations cover all 4 camera viewIds (not just view[0])
        observed_view_ids = {
            obs["observationId"]
            for lm in sfm_data["structure"]
            for obs in lm.get("observations", [])
        }
        self.assertEqual(len(observed_view_ids), 4)

        sfm_dense_data = json.loads((out_dir / "sfmDataDense.sfm").read_text(encoding="utf-8"))
        self.assertEqual(len(sfm_dense_data["views"]), 12)
        self.assertEqual(len(sfm_dense_data["poses"]), 12)
        dense_observed_view_ids = {
            obs["observationId"]
            for lm in sfm_dense_data.get("structure", [])
            for obs in lm.get("observations", [])
        }
        self.assertGreaterEqual(len(dense_observed_view_ids), 8)

    def test_cloud_api_polling_relative_artifact_and_error_cleanup(self) -> None:
        import io
        import zipfile
        import requests
        from unittest.mock import MagicMock, patch
        from gensfm_core.cloud_client import run_gensfm_cloud_api

        img_dir = self._create_synthetic_views(self.tmp_dir / "cloud_views", count=2)
        img_paths = sorted(img_dir.glob("*.png"))
        cloud_out = self.tmp_dir / "cloud_out"
        cloud_out.mkdir(parents=True, exist_ok=True)

        # Build a synthetic ZIP artifact bundle
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr(
                "cameras.json",
                json.dumps(
                    {
                        "extrinsics": [np.eye(4).tolist(), np.eye(4).tolist()],
                        "intrinsics": [np.eye(3).tolist(), np.eye(3).tolist()],
                        "data_to_mesh": DEFAULT_DATA_TO_MESH,
                    }
                ),
            )
            zf.writestr("mesh.glb", b"glTF_mock")
        zip_bytes = zip_buf.getvalue()

        # 1. Test POST /api/v1/jobs returning {"job_id": "j1"} (no initial status) -> GET polling -> relative artifact_url
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.headers = {"Content-Type": "application/json"}
        post_resp.content = b'{"job_id": "j1"}'
        post_resp.json.return_value = {"job_id": "j1"}

        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.headers = {"Content-Type": "application/json"}
        poll_resp.content = b'{"status": "completed"}'
        poll_resp.json.return_value = {
            "job_id": "j1",
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "artifact_url": "/api/v1/jobs/j1/artifacts",
        }

        art_resp = MagicMock()
        art_resp.status_code = 200
        art_resp.headers = {"Content-Type": "application/zip"}
        art_resp.content = zip_bytes

        with patch("requests.post", return_value=post_resp) as m_post, patch(
            "requests.get", side_effect=[poll_resp, art_resp]
        ) as m_get:
            run_gensfm_cloud_api(
                image_paths=img_paths,
                output_dir=cloud_out,
                endpoint_url="http://127.0.0.1:8000",
                api_key="secret_token",
                remove_background=True,
            )
            self.assertTrue((cloud_out / "cameras.json").is_file())
            self.assertTrue((cloud_out / "mesh.glb").is_file())
            self.assertEqual(m_post.call_args.kwargs["data"]["remove_background"], "true")
            self.assertEqual(m_get.call_args_list[0].args[0], "http://127.0.0.1:8000/api/v1/jobs/j1")
            self.assertEqual(
                m_get.call_args_list[1].args[0], "http://127.0.0.1:8000/api/v1/jobs/j1/artifacts"
            )

        # 2. Test HTTP error cleans up temporary .part files (PRD Section 7.1)
        part_file = cloud_out / "_cloud_partial.zip"
        part_file.write_bytes(b"partial")
        err_resp = MagicMock()
        err_resp.status_code = 401
        http_err = requests.exceptions.HTTPError("401 Unauthorized", response=err_resp)
        err_resp.raise_for_status.side_effect = http_err

        with patch("requests.post", return_value=err_resp):
            with self.assertRaises(requests.exceptions.HTTPError):
                run_gensfm_cloud_api(
                    image_paths=img_paths,
                    output_dir=cloud_out,
                    endpoint_url="http://127.0.0.1:8000",
                    api_key="bad_key",
                )
        self.assertFalse(part_file.exists())

    def test_downstream_meshroom_pipeline_templates_and_wagon_demo(self) -> None:
        # 1. Validate Streamlined 3-Node Template (genSfm.mg: CameraInit -> GenSfM -> Publish_Outputs)
        tpl_path = PLUGIN_ROOT / "meshroom/genSfm.mg"
        self.assertTrue(tpl_path.is_file(), f"Missing template: {tpl_path}")
        tpl = json.loads(tpl_path.read_text(encoding="utf-8"))
        self.assertTrue(tpl["header"].get("template", False))
        graph = tpl["graph"]
        for node_name in [
            "CameraInit_1",
            "GenSfM_1",
            "Publish_Outputs",
        ]:
            self.assertIn(node_name, graph, f"{node_name} missing in genSfm.mg")

        # Verify connections
        self.assertEqual(graph["GenSfM_1"]["inputs"]["input"], "{CameraInit_1.output}")
        self.assertEqual(graph["GenSfM_1"]["inputs"].get("maskingStrategy"), "auto_rembg")

        pub_files = graph["Publish_Outputs"]["inputs"]["inputFiles"]
        for expected_ref in [
            "{GenSfM_1.outputMeshObj}",
            "{GenSfM_1.outputMeshMtl}",
            "{GenSfM_1.outputMeshTexture}",
            "{GenSfM_1.outputMeshGlb}",
            "{GenSfM_1.outputGaussianPly}",
            "{GenSfM_1.outputSfMData}",
        ]:
            self.assertIn(expected_ref, pub_files)

        # 2. Validate GenSfM.py and GenSfMRunnerConfig default generateSyntheticViews == 24
        default_cfg = GenSfMRunnerConfig(input_path="dummy_in", output_dir="dummy_out")
        self.assertEqual(default_cfg.generate_synthetic_views, 24)
        gensfm_py_text = (PLUGIN_ROOT / "meshroom/genSfm/GenSfM.py").read_text(encoding="utf-8")
        self.assertIn('name="generateSyntheticViews"', gensfm_py_text)
        self.assertIn("value=24,", gensfm_py_text)

        # 3. Validate clean labels (no parenthesis tags)
        for output_tag in [
            'label="SfMData"',
            'label="Textured Mesh"',
            'label="Mesh Material"',
            'label="Mesh Texture"',
            'label="GLB Mesh"',
            'label="Gaussian Splats"',
            'label="Cameras"',
            'label="Dense SfMData"',
        ]:
            self.assertIn(output_tag, gensfm_py_text)


if __name__ == "__main__":
    unittest.main()
