from __future__ import annotations

import sys
import unittest
import base64
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from upscaler import ultimate_tile_rects, upscale_model_refine, upscale_ultimate  # noqa: E402


def _image_b64(size: tuple[int, int]) -> str:
    image = Image.new("RGB", size, (64, 96, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FakePipeline:
    def is_loaded(self) -> bool:
        return True

    def generate(self, req):
        return ([_image_b64((req.width, req.height))], 42, [], [], [{"seed": 42}])


class UltimateUpscaleTests(unittest.TestCase):
    def test_linear_tile_rects_cover_image(self) -> None:
        rects = ultimate_tile_rects(2300, 1400, tile=1024, mode="linear", seam_mode="none")

        self.assertEqual(rects[0], (0, 0, 1024, 1024))
        self.assertIn((2048, 1024, 2300, 1400), rects)
        self.assertEqual(len(rects), 6)

    def test_chess_tile_order_processes_even_before_odd(self) -> None:
        rects = ultimate_tile_rects(2048, 2048, tile=1024, mode="chess", seam_mode="none")

        self.assertEqual(rects[:2], [(0, 0, 1024, 1024), (1024, 1024, 2048, 2048)])
        self.assertEqual(rects[2:], [(1024, 0, 2048, 1024), (0, 1024, 1024, 2048)])

    def test_half_tile_intersections_adds_offset_pass(self) -> None:
        rects = ultimate_tile_rects(2048, 2048, tile=1024, mode="linear", seam_mode="half_tile_intersections")

        self.assertIn((512, 512, 1536, 1536), rects)
        self.assertGreater(len(rects), 4)

    def test_model_refine_accepts_generation_metadata_return(self) -> None:
        image = Image.new("RGB", (64, 64), (12, 34, 56))

        result = upscale_model_refine(image, FakePipeline(), tile_size=128)

        self.assertEqual(result.size, (64, 64))

    def test_ultimate_accepts_generation_metadata_return(self) -> None:
        image = Image.new("RGB", (64, 64), (12, 34, 56))

        result = upscale_ultimate(
            image,
            FakePipeline(),
            ROOT / "models",
            scale=1.0,
            tile=128,
            padding=0,
            mask_blur=0,
            steps=1,
            seam_mode="none",
        )

        self.assertEqual(result.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
