from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class OfficialExpansionTests(unittest.TestCase):
    def test_expander_uses_official_krea_expansion_rules(self) -> None:
        from prompt_expander import EXPANSION_SYSTEM_PROMPT

        # Phrases unique to Krea's official expansion.txt that our paraphrase lacked.
        self.assertIn("Faithfulness First", EXPANSION_SYSTEM_PROMPT)
        self.assertIn("Preserve User Medium", EXPANSION_SYSTEM_PROMPT)
        self.assertIn("two or three alternatives", EXPANSION_SYSTEM_PROMPT)
        self.assertIn("dignity", EXPANSION_SYSTEM_PROMPT)

    def test_expander_preserves_visible_text_without_inventing_copy(self) -> None:
        from prompt_expander import EXPANSION_SYSTEM_PROMPT

        self.assertIn("verbatim", EXPANSION_SYSTEM_PROMPT)
        self.assertIn("named object or surface", EXPANSION_SYSTEM_PROMPT)
        self.assertIn("Never invent additional readable text", EXPANSION_SYSTEM_PROMPT)


class EmotionRebalanceTests(unittest.TestCase):
    def test_emotion_preset_matches_community_weights(self) -> None:
        from conditioning import resolve_rebalance_weights

        weights = resolve_rebalance_weights("emotion")

        self.assertEqual(weights, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.75, 1.0, 1.1, 4.0, 1.0])

    def test_emotion_preset_is_gentler_than_balanced_on_deep_band(self) -> None:
        from conditioning import resolve_rebalance_weights

        emotion = resolve_rebalance_weights("emotion")
        balanced = resolve_rebalance_weights("balanced")

        # Deep-band (index 8) is far gentler than the default 5.0, reducing the
        # over-saturation / text damage the community reported.
        self.assertLess(emotion[8], balanced[8])


class RawProfileTests(unittest.TestCase):
    def test_raw_profile_uses_empty_negative_and_high_step_range(self) -> None:
        from model_profiles import MODEL_PROFILES

        raw = MODEL_PROFILES["krea_raw"]
        self.assertEqual(raw.default_cfg, 3.5)
        self.assertGreaterEqual(raw.default_steps, 40)
        self.assertLessEqual(raw.default_steps, 60)

    def test_turbo_shift_is_pinned_not_resolution_scaled(self) -> None:
        from model_profiles import MODEL_PROFILES

        # Pinned mu => constant time shift, independent of resolution (Kijai).
        self.assertEqual(MODEL_PROFILES["krea_turbo"].default_mu, 1.15)
        # RAW uses the documented default (resolution-aware) sampling.
        self.assertIsNone(MODEL_PROFILES["krea_raw"].default_mu)


class OfficialAssetTests(unittest.TestCase):
    def test_turbo_distill_lora_is_registered(self) -> None:
        from lora_manager import OFFICIAL_LORAS

        self.assertIn("krea2_turbo_lora_rank_64_bf16", OFFICIAL_LORAS)

    def test_raw_fp8_checkpoint_is_a_quality_asset(self) -> None:
        from quality_assets import asset_specs

        ids = {spec.id for spec in asset_specs()}
        self.assertIn("krea2_raw_fp8", ids)


class WarmPastelLoraTests(unittest.TestCase):
    def test_warmpastel_lora_registered_with_trigger(self) -> None:
        from lora_manager import OFFICIAL_LORAS, OFFICIAL_LORA_HF_IDS

        self.assertIn("krea2_warmpastel", OFFICIAL_LORAS)
        self.assertTrue(OFFICIAL_LORAS["krea2_warmpastel"]["trigger_words"])
        self.assertEqual(OFFICIAL_LORA_HF_IDS["krea2_warmpastel"], "Comfy-Org/Krea-2")


class PromptingGuideTests(unittest.TestCase):
    def test_official_prompting_guide_has_key_tips(self) -> None:
        from prompting_guide import OFFICIAL_PROMPTING_GUIDE, prompting_guide_payload

        text = OFFICIAL_PROMPTING_GUIDE.lower()
        self.assertIn("natural language", text)
        self.assertIn("quotes", text)  # text rendering tip
        self.assertIn("2k", text)
        payload = prompting_guide_payload()
        self.assertIn("guidelines", payload)
        self.assertTrue(payload["examples"])

    def test_heuristic_planner_follows_official_tips_for_text(self) -> None:
        from prompt_planner import plan_prompt_heuristic

        result = plan_prompt_heuristic('a sign that says "OPEN"')
        # Text-rendering requests keep the quoted words intact.
        self.assertIn('"OPEN"', result.planned_prompt)


class AltVaeTests(unittest.TestCase):
    def test_alt_vae_assets_registered(self) -> None:
        from quality_assets import asset_specs

        ids = {spec.id for spec in asset_specs()}
        self.assertIn("qwen_image_hdr_vae", ids)


class Fp16ModeTests(unittest.TestCase):
    def test_generation_request_accepts_fp16(self) -> None:
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="x", quantization="fp16")
        self.assertEqual(req.quantization, "fp16")


if __name__ == "__main__":
    unittest.main()
