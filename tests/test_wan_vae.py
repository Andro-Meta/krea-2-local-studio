from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class WanVaeTests(unittest.TestCase):
    def test_detects_wan_style_vae_keys(self) -> None:
        from krea2.wan_vae import is_wan_vae_state_dict

        self.assertTrue(is_wan_vae_state_dict({
            "encoder.conv1.weight": object(),
            "decoder.conv1.weight": object(),
            "conv1.weight": object(),
            "conv2.weight": object(),
        }))
        self.assertFalse(is_wan_vae_state_dict({"encoder.conv_in.weight": object()}))

    def test_qwen_autoencoder_uses_wan_override_when_keys_match(self) -> None:
        from krea2.autoencoder import QwenAutoencoder

        fake_ae = Mock()
        fake_ae.config = type("FakeConfig", (), {
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "scaling_factor": None,
        })()
        fake_wan = Mock()
        fake_wan.config = type("FakeWanConfig", (), {
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "scaling_factor": None,
        })()
        fake_wan.load_state_dict.return_value = None
        fake_wan.requires_grad_.return_value = fake_wan
        fake_wan.upscale_factor = 1

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wan_2.1_vae.safetensors"
            path.write_bytes(b"stub")
            with (
                patch("diffusers.AutoencoderKLQwenImage.from_pretrained", return_value=fake_ae),
                patch("support_models.support_model_path", return_value="unused"),
                patch("safetensors.torch.load_file", return_value={
                    "encoder.conv1.weight": object(),
                    "decoder.conv1.weight": object(),
                    "conv1.weight": object(),
                    "conv2.weight": object(),
                }),
                patch("krea2.autoencoder.WanAutoencoder", return_value=fake_wan),
            ):
                ae = QwenAutoencoder(str(path), vae_mode="wan_experimental")

        self.assertIs(ae.ae, fake_wan)
        self.assertTrue(ae.vae_source.startswith("wan_experimental:"))

    def test_qwen_autoencoder_labels_comfy_qwen_vae(self) -> None:
        from krea2.autoencoder import QwenAutoencoder

        fake_ae = Mock()
        fake_ae.config = type("FakeConfig", (), {
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "scaling_factor": None,
        })()
        fake_qwen = Mock()
        fake_qwen.config = fake_ae.config
        fake_qwen.requires_grad_.return_value = fake_qwen
        fake_qwen.upscale_factor = 1

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen_image_vae.safetensors"
            path.write_bytes(b"stub")
            with (
                patch("diffusers.AutoencoderKLQwenImage.from_pretrained", return_value=fake_ae),
                patch("support_models.support_model_path", return_value="unused"),
                patch("safetensors.torch.load_file", return_value={
                    "encoder.conv1.weight": object(),
                    "decoder.conv1.weight": object(),
                    "conv1.weight": object(),
                    "conv2.weight": object(),
                }),
                patch("krea2.autoencoder.WanAutoencoder", return_value=fake_qwen),
            ):
                ae = QwenAutoencoder(str(path), vae_mode="comfy_qwen")

        self.assertIs(ae.ae, fake_qwen)
        self.assertTrue(ae.vae_source.startswith("comfy_qwen:"))

    def test_laplacian_detail_blend_preserves_shape_and_range(self) -> None:
        from krea2.autoencoder import _laplacian_detail_blend

        low = torch.zeros(1, 3, 32, 32)
        high = torch.ones(1, 3, 32, 32) * 0.5

        out = _laplacian_detail_blend(low, high, blur_radius=4, high_strength=0.7)

        self.assertEqual(tuple(out.shape), tuple(low.shape))
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreaterEqual(float(out.min()), -1.0)

    def test_spacepxl_2x_decoder_is_detected_from_output_channels(self) -> None:
        import krea2.wan_vae as wan_vae

        class FakeWanVAE:
            def load_state_dict(self, _state_dict, strict=True):
                return None

        with patch.object(wan_vae, "WanVAE2D", return_value=FakeWanVAE()):
            ae = wan_vae.WanAutoencoder({"decoder.head.2.weight": torch.zeros(12, 96, 3, 3, 3)})

        self.assertEqual(ae.upscale_factor, 2)


if __name__ == "__main__":
    unittest.main()
