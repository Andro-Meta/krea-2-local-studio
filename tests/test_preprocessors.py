from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from preprocessors import pixel_perfect_resolution, preprocess_image  # noqa: E402


class PreprocessorTests(unittest.TestCase):
    def test_pixel_perfect_resolution_preserves_aspect_and_aligns_to_8(self) -> None:
        width, height = pixel_perfect_resolution(1600, 900, 768)

        self.assertEqual(width % 8, 0)
        self.assertEqual(height % 8, 0)
        self.assertLessEqual(max(width, height), 768)
        self.assertGreater(width, height)

    def test_canny_preview_returns_rgb_at_target_size(self) -> None:
        image = Image.new("RGB", (1024, 512), "white")

        preview = preprocess_image(image, kind="canny", resolution=512)

        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(preview.size, (512, 256))

    def test_unknown_preprocessor_fails_clearly(self) -> None:
        image = Image.new("RGB", (64, 64), "white")

        with self.assertRaisesRegex(ValueError, "Unknown preprocessor"):
            preprocess_image(image, kind="sam3")


if __name__ == "__main__":
    unittest.main()
