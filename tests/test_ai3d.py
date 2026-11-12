"""
Unit & Integration Tests for mrAIGen3D Plugin.
"""
import unittest
from pathlib import Path
import sys

plugin_root = Path(__file__).resolve().parents[1]
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from mrai3d_core.sfm_helper import resolve_single_image
from mrai3d_core.wsl_bridge import run_ai3d_in_wsl, is_wsl_available


class TestAIGen3D(unittest.TestCase):

    def setUp(self):
        self.test_img = Path("D:/Meshroom/griffin/1.png")
        self.output_dir = Path("D:/Meshroom/mrAIGen3D/tests/unit_test_out")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_01_resolve_single_image(self):
        img_p, v_id = resolve_single_image(
            input_sfm_str=str(self.test_img),
            selected_view_index=0,
        )
        self.assertTrue(img_p.is_file())
        self.assertEqual(v_id, "0")

    def test_02_resolve_from_directory(self):
        img_p, v_id = resolve_single_image(
            input_sfm_str="D:/Meshroom/griffin",
            selected_view_index=2,
        )
        self.assertTrue(img_p.is_file())
        self.assertEqual(v_id, "2")

    def test_03_triposr_wsl_execution(self):
        if not is_wsl_available():
            self.skipTest("WSL2 not available")

        out_obj = self.output_dir / "mesh.obj"
        out_glb = self.output_dir / "asset_pbr.glb"
        out_ply = self.output_dir / "gaussian.ply"

        returncode = run_ai3d_in_wsl(
            image_path=self.test_img,
            output_obj=out_obj,
            output_glb=out_glb,
            output_ply=out_ply,
            model_name="triposr",
            mc_resolution=192,
            chunk_size=4096,
        )
        self.assertEqual(returncode, 0)
        self.assertTrue(out_obj.is_file(), f"Expected OBJ file {out_obj}")
        self.assertTrue(out_glb.is_file(), f"Expected GLB file {out_glb}")

        # Check non-empty content
        self.assertGreater(out_obj.stat().st_size, 1000)
        self.assertGreater(out_glb.stat().st_size, 1000)

    def test_04_colab_bridge_client_mock(self):
        import http.server
        import threading
        import io
        import zipfile
        import json
        from mrai3d_core.cloud_bridge import ColabBridgeClient

        class MockHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "gpu": "Mock Tesla T4", "vram_total_mb": 15360}).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/generate":
                    content_length = int(self.headers.get("Content-Length", 0))
                    _ = self.rfile.read(content_length)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.end_headers()
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w") as z:
                        z.writestr("mesh.obj", "# mock obj")
                        z.writestr("asset_pbr.glb", b"glTFmock")
                    self.wfile.write(buf.getvalue())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), MockHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever)
        t.daemon = True
        t.start()

        try:
            client = ColabBridgeClient(f"http://127.0.0.1:{port}")
            health = client.check_health()
            self.assertEqual(health["gpu"], "Mock Tesla T4")

            colab_out = self.output_dir / "colab_mock_out"
            obj_p, glb_p = client.generate(image_path=self.test_img, output_dir=colab_out)
            self.assertTrue(obj_p.is_file())
            self.assertTrue(glb_p.is_file())
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

