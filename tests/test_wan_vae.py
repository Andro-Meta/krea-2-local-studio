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
                ae = QwenAutoencoder(str(path))

        self.assertIs(ae.ae, fake_wan)
        self.assertTrue(ae.vae_source.startswith("wan2.1:"))


if __name__ == "__main__":
    unittest.main()
