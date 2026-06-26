from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class QualityUpgradeTests(unittest.TestCase):
    def test_provider_auto_uses_krea_when_flux_missing(self) -> None:
        import edit_providers

        provider = edit_providers.resolve_edit_provider("auto", "inpaint", flux_fill_installed=False)
        self.assertEqual(provider.name, "krea_native")
        self.assertIn("FLUX Fill", provider.reason)

    def test_provider_auto_uses_flux_for_strict_edits_when_available(self) -> None:
        import edit_providers

        provider = edit_providers.resolve_edit_provider("auto", "outpaint", flux_fill_installed=True)
        self.assertEqual(provider.name, "flux_fill")

    def test_mask_crop_expands_and_composites_back(self) -> None:
        import mask_editing

        image = Image.new("RGB", (128, 128), "navy")
        mask = Image.new("L", (128, 128), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((54, 50, 74, 72), fill=255)

        crop = mask_editing.crop_for_mask(image, mask, padding=16, align=16)
        self.assertEqual(crop.box, (32, 32, 96, 96))
        self.assertEqual(crop.image.size, (64, 64))
        self.assertEqual(crop.mask.size, (64, 64))

        generated = Image.new("RGB", crop.image.size, "orange")
        composited = mask_editing.composite_crop(image, generated, crop.feathered_mask, crop.box)
        self.assertEqual(composited.size, image.size)
        self.assertNotEqual(composited.getpixel((64, 60)), image.getpixel((64, 60)))
        self.assertEqual(composited.getpixel((5, 5)), image.getpixel((5, 5)))

    def test_output_encoder_can_skip_disk_writes_for_previews(self) -> None:
        import output_saver

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            image = Image.new("RGB", (16, 16), "purple")
            encoded, filenames = output_saver.encode_images([image], out_dir, save_outputs=False)
            saved_files = list(out_dir.glob("*.png"))

        self.assertEqual(len(encoded), 1)
        self.assertEqual(filenames, [])
        self.assertEqual(saved_files, [])

    def test_official_lora_download_uses_loras_subfolder(self) -> None:
        import lora_manager

        kwargs = lora_manager.official_lora_download_kwargs("krea2_darkbrush")

        self.assertEqual(kwargs["repo_id"], "Comfy-Org/Krea-2")
        self.assertEqual(kwargs["filename"], "krea2_darkbrush.safetensors")
        self.assertEqual(kwargs["subfolder"], "loras")


if __name__ == "__main__":
    unittest.main()
