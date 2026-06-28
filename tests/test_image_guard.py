from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("numpy/PIL required for image guard tests") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ImageGuardTests(unittest.TestCase):
    def test_black_image_is_flagged(self) -> None:
        from image_guard import assess_image

        rep = assess_image(Image.new("RGB", (64, 64), (0, 0, 0)))
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["issue"], "black")

    def test_uniform_nonblack_image_is_flagged(self) -> None:
        from image_guard import assess_image

        rep = assess_image(Image.new("RGB", (64, 64), (130, 130, 130)))
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["issue"], "uniform")

    def test_natural_image_passes(self) -> None:
        from image_guard import assess_image

        rng = np.random.default_rng(0)
        arr = (rng.random((64, 64, 3)) * 255).astype("uint8")
        rep = assess_image(Image.fromarray(arr))
        self.assertTrue(rep["ok"])
        self.assertIsNone(rep["issue"])

    def test_nan_array_is_flagged(self) -> None:
        from image_guard import assess_image_array

        arr = np.full((8, 8, 3), np.nan, dtype="float32")
        rep = assess_image_array(arr)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["issue"], "nan")

    def test_summarize_marks_all_bad_vs_some_bad(self) -> None:
        from image_guard import assess_image, summarize_quality

        black = Image.new("RGB", (32, 32), (0, 0, 0))
        rng = np.random.default_rng(1)
        good = Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))

        all_bad = summarize_quality([assess_image(black), assess_image(black)])
        some_bad = summarize_quality([assess_image(black), assess_image(good)])

        self.assertTrue(all_bad["all_bad"])
        self.assertFalse(some_bad["all_bad"])
        self.assertEqual(some_bad["bad_count"], 1)


if __name__ == "__main__":
    unittest.main()
