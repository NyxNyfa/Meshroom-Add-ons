"""
Unit & Integration Tests for mrRemBG Plugin.
"""
import json
import unittest
from pathlib import Path
from PIL import Image

import sys
plugin_root = Path(__file__).resolve().parents[1]
if str(plugin_root) not in sys.path:
    sys.path.insert(0, str(plugin_root))

from rembg_core.sfm_io import SfmDataIO
from rembg_core.wsl_bridge import run_rembg_in_wsl, is_wsl_available


class TestRemBG(unittest.TestCase):

    def setUp(self):
        self.test_img = Path("D:/Meshroom/griffin/1.png")
        self.output_dir = Path("D:/Meshroom/mrRemBG/tests/unit_test_out")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_01_sfm_io_load_single_image(self):
        io_handler = SfmDataIO()
        views, orig_dict = io_handler.load_views(str(self.test_img))
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].image_path.resolve(), self.test_img.resolve())
        self.assertGreater(views[0].width, 0)
        self.assertGreater(views[0].height, 0)

    def test_02_sfm_io_load_directory(self):
        io_handler = SfmDataIO()
        views, _ = io_handler.load_views("D:/Meshroom/griffin")
        self.assertGreaterEqual(len(views), 4)

    def test_03_wsl_execution_and_outputs(self):
        if not is_wsl_available():
            self.skipTest("WSL2 not available")

        rgba_dir = self.output_dir / "rgba"
        mask_dir = self.output_dir / "mask"
        out_sfm = self.output_dir / "sfmData.sfm"

        returncode = run_rembg_in_wsl(
            input_paths=[self.test_img],
            output_rgba_dir=rgba_dir,
            output_mask_dir=mask_dir,
            output_sfm_path=out_sfm,
            model_name="u2net",
            pad_to_square=True,
        )
        self.assertEqual(returncode, 0)

        # Verify RGBA output
        rgba_file = rgba_dir / "1_rgba.png"
        self.assertTrue(rgba_file.is_file(), f"Expected RGBA file {rgba_file}")
        with Image.open(rgba_file) as im:
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.size[0], im.size[1], "pad_to_square should produce 1:1 image")

        # Verify mask output
        mask_file = mask_dir / "1_mask.png"
        self.assertTrue(mask_file.is_file(), f"Expected mask file {mask_file}")
        with Image.open(mask_file) as im:
            self.assertIn(im.mode, ("L", "1"))

        # Verify sfmData output
        self.assertTrue(out_sfm.is_file())
        data = json.loads(out_sfm.read_text(encoding="utf-8"))
        self.assertIn("views", data)
        self.assertEqual(len(data["views"]), 1)
        self.assertTrue(Path(data["views"][0]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()

