from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from inference import _mask_to_tensor, _outpaint_seam_mask, _outpaint_stitch


class OutpaintMaskTests(unittest.TestCase):
    def test_outpaint_mask_preserves_feather_values(self) -> None:
        img = Image.new("L", (64, 64), 0)
        for x in range(30, 64):
            for y in range(64):
                img.putpixel((x, y), 255)

        outpaint = _mask_to_tensor(img, 64, 64, "outpaint", "cpu", torch.float32)
        inpaint = _mask_to_tensor(img, 64, 64, "inpaint", "cpu", torch.float32)

        outpaint_values = torch.unique(outpaint)
        inpaint_values = torch.unique(inpaint)

        self.assertGreater(len(outpaint_values), len(inpaint_values))
        self.assertTrue(torch.any((outpaint_values > 0) & (outpaint_values < 1)))
        self.assertFalse(torch.any((inpaint_values > 0) & (inpaint_values < 1)))

    def test_outpaint_stitch_preserves_unmasked_source(self) -> None:
        base = Image.new("RGB", (16, 16), (10, 20, 30))
        generated = Image.new("RGB", (16, 16), (200, 210, 220))
        mask = Image.new("L", (16, 16), 255)
        for x in range(8):
            for y in range(16):
                mask.putpixel((x, y), 0)

        stitched = _outpaint_stitch([generated], base, mask)[0]

        self.assertEqual(stitched.getpixel((2, 8)), (10, 20, 30))
        self.assertEqual(stitched.getpixel((14, 8)), (200, 210, 220))

    def test_outpaint_seam_mask_targets_feather_band(self) -> None:
        mask = Image.new("L", (64, 16), 255)
        mask.paste(0, (0, 0, 32, 16))
        for x in range(16):
            value = int(255 * x / 15)
            for y in range(16):
                mask.putpixel((24 + x, y), value)

        seam = _outpaint_seam_mask(mask, (64, 16))

        self.assertIsNotNone(seam.getbbox())
        self.assertGreater(seam.getpixel((32, 8)), 0)
        self.assertLess(seam.getpixel((4, 8)), seam.getpixel((32, 8)))


if __name__ == "__main__":
    unittest.main()
