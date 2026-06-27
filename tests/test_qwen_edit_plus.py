from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2.edit_plus import (  # noqa: E402
    EDIT_PLUS_IMAGE_SIZE,
    resize_edit_plus_images,
    wrap_image_edit_plus_prompt,
)


class QwenEditPlusTests(unittest.TestCase):
    def test_edit_plus_prompt_caps_to_three_images(self) -> None:
        prompt = wrap_image_edit_plus_prompt("replace the sign text", 5)

        self.assertEqual(prompt.count("<|image_pad|>"), 3)
        self.assertIn("Picture 1:", prompt)
        self.assertIn("Picture 3:", prompt)
        self.assertIn("replace the sign text", prompt)

    def test_edit_plus_resizes_semantic_images_to_384_square(self) -> None:
        images = [
            Image.new("RGB", (1024, 512), "red"),
            Image.new("RGB", (128, 2048), "blue"),
            Image.new("RGB", (512, 512), "green"),
            Image.new("RGB", (64, 64), "black"),
        ]

        resized = resize_edit_plus_images(images)

        self.assertEqual(len(resized), 3)
        self.assertTrue(all(image.size == (EDIT_PLUS_IMAGE_SIZE, EDIT_PLUS_IMAGE_SIZE) for image in resized))


if __name__ == "__main__":
    unittest.main()
