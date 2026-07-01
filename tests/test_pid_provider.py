from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class PiDProviderTests(unittest.TestCase):
    def test_quality_assets_include_pid_decoder_and_text_encoder(self) -> None:
        import quality_assets

        specs = {spec.id: spec for spec in quality_assets.asset_specs()}

        self.assertEqual(specs["pid_qwenimage_decoder"].repo_id, "Comfy-Org/PixelDiT")
        self.assertEqual(specs["pid_qwenimage_decoder"].filename, "diffusion_models/pid_qwenimage_1024_to_4096_4step_bf16.safetensors")
        self.assertEqual(specs["pid_gemma_text_encoder"].filename, "text_encoders/gemma_2_2b_it_elm_bf16.safetensors")
        self.assertEqual(specs["pid_qwenimage_official_checkpoint"].repo_id, "nvidia/PiD")
        self.assertEqual(specs["pid_qwenimage_vae_2d"].filename, "checkpoints/QwenImage_VAE_2d.pth")

    def test_status_blocks_sageattention_and_missing_assets(self) -> None:
        from pid_decoder_provider import PiDSettings, pid_status

        with patch("pid_decoder_provider.accelerator_status", return_value={"sageattention": {"selected": True, "installed": True}}):
            status = pid_status(PiDSettings(decoder_path="", text_encoder_path=""), free_vram_gb=24.0)

        self.assertFalse(status["available"])
        self.assertIn("SageAttention", " ".join(status["blocked_reasons"]))
        self.assertIn("PiD Comfy decoder", " ".join(status["blocked_reasons"]))

    def test_status_allows_installed_assets_with_enough_vram(self) -> None:
        from pid_decoder_provider import PiDSettings, pid_status

        with (
            patch("pid_decoder_provider.Path.is_file", return_value=True),
            patch("pid_decoder_provider.accelerator_status", return_value={"sageattention": {"selected": False, "installed": False}}),
        ):
            status = pid_status(
                PiDSettings(
                    decoder_path="decoder.safetensors",
                    text_encoder_path="gemma.safetensors",
                    official_checkpoint_path="model_ema_bf16.pth",
                    official_vae_path="QwenImage_VAE_2d.pth",
                ),
                free_vram_gb=18.0,
            )

        self.assertTrue(status["available"])
        self.assertEqual(status["estimated_vram_gb"], 15.0)

    def test_pid_upscale_refuses_without_runtime(self) -> None:
        from PIL import Image

        from pid_decoder_provider import PiDSettings, upscale_pid

        with (
            patch("pid_decoder_provider.Path.is_file", return_value=True),
            patch("pid_decoder_provider.accelerator_status", return_value={"sageattention": {"selected": False}}),
            self.assertRaisesRegex(RuntimeError, "Official PiD runtime checkpoint|PiD runtime"),
        ):
            upscale_pid(Image.new("RGB", (16, 16), "black"), PiDSettings(decoder_path="x", text_encoder_path="y"))

    def test_api_status_shape(self) -> None:
        from fastapi.testclient import TestClient
        import main

        with TestClient(main.app) as client:
            response = client.get("/api/pid/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("available", data)
        self.assertIn("assets", data)
        self.assertIn("blocked_reasons", data)

    def test_pid_is_not_default_when_unavailable(self) -> None:
        import settings

        self.assertFalse(hasattr(settings.settings, "krea2_default_postprocess"))


if __name__ == "__main__":
    unittest.main()
