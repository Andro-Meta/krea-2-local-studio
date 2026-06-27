from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from upscaler import ultimate_tile_rects  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
