from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2.reference_image import (  # noqa: E402
    wrap_image_prompt,
    cap_vision_megapixels,
    crop_image_to_mask,
)


class Krea2ReferenceEncoderTests(unittest.TestCase):
    def test_mask_crop_uses_bbox_with_padding(self) -> None:
        image = Image.new("RGB", (100, 80), "black")
        mask = Image.new("L", (100, 80), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((40, 20, 59, 39), fill=255)

        cropped = crop_image_to_mask(image, mask, padding=10)

        self.assertEqual(cropped.size, (40, 40))

    def test_vision_megapixel_cap_preserves_aspect_without_upscale(self) -> None:
        large = Image.new("RGB", (2000, 1000), "white")
        small = Image.new("RGB", (200, 100), "white")

        capped = cap_vision_megapixels(large, 0.5)
        unchanged = cap_vision_megapixels(small, 0.5)

        self.assertEqual(capped.size, (1000, 500))
        self.assertEqual(unchanged.size, (200, 100))

    def test_prompt_wrapper_supports_image_after_prompt(self) -> None:
        wrapped = wrap_image_prompt("paint a red jacket", 1, vision_position="after_prompt")

        self.assertLess(wrapped.index("paint a red jacket"), wrapped.index("<|image_pad|>"))

    def test_prompt_wrapper_bounds_system_prompt_override(self) -> None:
        wrapped = wrap_image_prompt("x", 1, system_prompt="a" * 1000)

        self.assertIn("a" * 512, wrapped)
        self.assertNotIn("a" * 513, wrapped)


if __name__ == "__main__":
    unittest.main()
