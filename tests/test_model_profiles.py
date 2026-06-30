from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from model_profiles import apply_profile_defaults, engine_catalog, model_profile_options, resolve_model_profile  # noqa: E402
from schemas import GenerationRequest  # noqa: E402
from support_models import support_model_status  # noqa: E402


class ModelProfileTests(unittest.TestCase):
    def test_krea_turbo_profile_applies_defaults(self) -> None:
        req = GenerationRequest(prompt="profile test", model_profile="krea_turbo")

        profile = apply_profile_defaults(req)

        self.assertEqual(profile.id, "krea_turbo")
        self.assertEqual(req.checkpoint, "turbo")
        self.assertEqual(req.quantization, "fp8")
        self.assertEqual(req.sampler, "euler")
        self.assertEqual(req.scheduler, "simple")
        self.assertEqual(req.steps, 8)
        self.assertEqual(req.cfg, 1.0)

    def test_krea_raw_profile_applies_raw_defaults(self) -> None:
        req = GenerationRequest(prompt="profile test", model_profile="krea_raw")

        apply_profile_defaults(req)

        self.assertEqual(req.checkpoint, "raw")
        self.assertEqual(req.quantization, "bf16")
        self.assertEqual(req.steps, 52)
        self.assertEqual(req.cfg, 3.5)
        self.assertIsNone(req.mu)

    def test_disabled_future_profiles_are_visible_but_blocked(self) -> None:
        options = model_profile_options()
        self.assertIn("z_image_turbo", [option["id"] for option in options])
        self.assertFalse(next(option for option in options if option["id"] == "z_image_turbo")["enabled"])

        req = GenerationRequest(prompt="profile test", model_profile="z_image_turbo")
        with self.assertRaisesRegex(ValueError, "not enabled"):
            apply_profile_defaults(req)

    def test_unknown_profile_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model profile"):
            resolve_model_profile("nope")

    def test_optional_family_assets_are_status_visible_but_download_gated(self) -> None:
        status = support_model_status()
        z_image = next(item for item in status if item["id"] == "z_image_turbo")

        self.assertTrue(z_image["optional"])
        self.assertFalse(z_image["download_enabled"])
        self.assertIn("Loader path", z_image["disabled_reason"])

    def test_engine_catalog_exposes_native_and_gguf_capabilities(self) -> None:
        catalog = engine_catalog()
        engines = {item["engine_id"]: item for item in catalog["engines"]}

        self.assertIn("native_pytorch", engines)
        self.assertIn("gguf_external", engines)
        self.assertIn("native_int8_convrot", engines)
        self.assertTrue(engines["native_pytorch"]["supports_lora"])
        self.assertTrue(engines["native_pytorch"]["supports_moodboards"])
        self.assertFalse(engines["gguf_external"]["supports_moodboards"])
        self.assertFalse(engines["gguf_external"]["supports_krea_enhancer"])
        self.assertEqual(engines["gguf_external"]["recommended_steps"], 8)
        self.assertTrue(engines["native_int8_convrot"]["supports_moodboards"])
        self.assertTrue(engines["native_int8_convrot"]["supports_realtime"])
        self.assertEqual(engines["native_int8_convrot"]["quantization"], "int8")
        self.assertNotIn("sidecar", engines["native_int8_convrot"]["label"].lower())


if __name__ == "__main__":
    unittest.main()
