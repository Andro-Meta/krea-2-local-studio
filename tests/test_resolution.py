from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ResolutionTests(unittest.TestCase):
    def test_square_tiers_match_long_side(self) -> None:
        from resolution import compute_dimensions

        self.assertEqual(compute_dimensions("1:1", "1k"), (1024, 1024))
        self.assertEqual(compute_dimensions("1:1", "2k"), (2048, 2048))

    def test_landscape_uses_long_side_as_width(self) -> None:
        from resolution import compute_dimensions

        self.assertEqual(compute_dimensions("16:9", "2k"), (2048, 1152))
        self.assertEqual(compute_dimensions("16:9", "1k"), (1024, 576))
        self.assertEqual(compute_dimensions("4:3", "2k"), (2048, 1536))

    def test_portrait_uses_long_side_as_height(self) -> None:
        from resolution import compute_dimensions

        self.assertEqual(compute_dimensions("9:16", "2k"), (1152, 2048))
        # 1024 * 2/3 = 682.67 -> nearest multiple of 16 = 688
        self.assertEqual(compute_dimensions("2:3", "1k"), (688, 1024))

    def test_all_outputs_are_16_aligned_and_capped(self) -> None:
        from resolution import ASPECT_RATIOS, RESOLUTION_TIERS, compute_dimensions

        for tier in RESOLUTION_TIERS:
            for aspect in ASPECT_RATIOS:
                w, h = compute_dimensions(aspect, tier)
                self.assertEqual(w % 16, 0, f"{aspect}/{tier} w={w}")
                self.assertEqual(h % 16, 0, f"{aspect}/{tier} h={h}")
                self.assertLessEqual(max(w, h), 2048)
                self.assertGreaterEqual(min(w, h), 256)
                # long side equals the tier target
                self.assertEqual(max(w, h), RESOLUTION_TIERS[tier])

    def test_unknown_aspect_or_tier_falls_back_to_square(self) -> None:
        from resolution import compute_dimensions

        self.assertEqual(compute_dimensions("bogus", "1k"), (1024, 1024))
        self.assertEqual(compute_dimensions("1:1", "bogus"), (1024, 1024))

    def test_normalize_dimensions_aligns_and_clamps(self) -> None:
        from resolution import normalize_dimensions

        self.assertEqual(normalize_dimensions(1000, 700), (992, 704))    # nearest /16
        self.assertEqual(normalize_dimensions(5000, 5000, max_edge=2048), (2048, 2048))
        self.assertEqual(normalize_dimensions(10, 10), (256, 256))       # floor


if __name__ == "__main__":
    unittest.main()
