from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboard_enrichment import (  # noqa: E402
    MoodboardSource,
    generate_moodboard_guidance,
    parse_qwen_guidance_json,
    sanitize_transferable_guidance,
)


class MoodboardEnrichmentTests(unittest.TestCase):
    def test_parses_qwen_guidance_from_fenced_json(self) -> None:
        text = """
        ```json
        {
          "prompt_guidance": "Use low-key practical lighting and tactile street texture.",
          "negative_guidance": "Avoid polished studio gloss.",
          "style_axes": ["gritty realism", "shallow depth of field"],
          "conditioning_notes": ["Favor close candid framing."],
          "source_summary": "Gritty cinematic realism translated for local generation."
        }
        ```
        """

        guidance = parse_qwen_guidance_json(text)

        self.assertEqual(guidance["guidance_version"], 1)
        self.assertIn("low-key", guidance["prompt_guidance"])
        self.assertEqual(guidance["style_axes"], ["gritty realism", "shallow depth of field"])

    def test_official_guidance_does_not_accept_reauthored_catalog_metadata(self) -> None:
        source = MoodboardSource(
            title="Gritty Cinematic Realism",
            taste_profile="Somber urban documentary suspense.",
            keywords=["cinematic realism"],
        )

        guidance = generate_moodboard_guidance(
            [source],
            mode="official",
            generator=lambda _prompt, _images: """
            {
              "title": "Different Title",
              "taste_profile": "Different profile.",
              "keywords": ["rewritten"],
              "prompt_guidance": "Use gritty realism and natural light.",
              "negative_guidance": "Avoid glossy fantasy.",
              "style_axes": ["documentary realism"],
              "conditioning_notes": ["Use source images as texture anchors."],
              "source_summary": "Official board guidance."
            }
            """,
        )

        self.assertNotIn("title", guidance)
        self.assertNotIn("taste_profile", guidance)
        self.assertNotIn("keywords", guidance)
        self.assertIn("gritty realism", guidance["prompt_guidance"])

    def test_guidance_prompt_requires_transferable_style_not_source_subject(self) -> None:
        prompt = __import__("moodboard_enrichment").build_moodboard_guidance_prompt(
            [
                MoodboardSource(
                    title="Analog Apocalyptic Radiance",
                    taste_profile="High-contrast thermal light and analog film grain.",
                    keywords=["analog film grain", "fiery orange glow", "high-contrast silhouette"],
                )
            ],
            "official",
        )

        self.assertIn("transferable visual style", prompt)
        self.assertIn("Do not describe the source image subject", prompt)
        self.assertIn("people-count", prompt)

    def test_custom_guidance_can_author_missing_catalog_metadata(self) -> None:
        source = MoodboardSource(
            title="",
            taste_profile="",
            keywords=[],
            image_b64s=["abc123"],
        )

        guidance = generate_moodboard_guidance(
            [source],
            mode="custom",
            generator=lambda _prompt, images: f"""
            {{
              "title": "Neon Rain Glass",
              "taste_profile": "A reflective cyber-noir style with rain-slick glass and pink rim light.",
              "keywords": ["cyber-noir", "rain glass", "pink rim light"],
              "prompt_guidance": "Use reflective wet surfaces and neon contrast. Images: {len(images)}.",
              "negative_guidance": "Avoid flat daylight.",
              "style_axes": ["neon noir"],
              "conditioning_notes": ["Use uploaded references for palette."],
              "source_summary": "Custom upload authored by Qwen."
            }}
            """,
        )

        self.assertEqual(guidance["title"], "Neon Rain Glass")
        self.assertIn("cyber-noir", guidance["keywords"])
        self.assertIn("Images: 1", guidance["prompt_guidance"])

    def test_invalid_qwen_response_falls_back_to_structured_guidance(self) -> None:
        source = MoodboardSource(
            title="Gritty Cinematic Realism",
            taste_profile="Somber urban documentary suspense.",
            keywords=["cinematic realism", "tactile texture"],
        )

        guidance = generate_moodboard_guidance(
            [source],
            mode="official",
            generator=lambda _prompt, _images: "not json at all",
        )

        self.assertIn("Gritty Cinematic Realism", guidance["prompt_guidance"])
        self.assertIn("cinematic realism", guidance["style_axes"])
        self.assertEqual(guidance["guidance_backend"], "heuristic_fallback")

    def test_sanitizes_subject_locking_guidance(self) -> None:
        guidance = sanitize_transferable_guidance({
            "prompt_guidance": "A lone figure centered in frame with fiery orange glow and analog film grain.",
            "negative_guidance": "Avoid clean digital rendering. Avoid crowds, text, buildings, subjects, or populated scenes. Avoid flat lighting.",
            "style_axes": ["analog film grain", "apocalyptic desolation"],
            "conditioning_notes": [],
            "source_summary": "",
            "guidance_version": 1,
        })

        self.assertNotIn("lone figure", guidance["prompt_guidance"].lower())
        self.assertNotIn("crowds", guidance["negative_guidance"].lower())
        self.assertNotIn("buildings", guidance["negative_guidance"].lower())
        self.assertNotIn("populated", guidance["negative_guidance"].lower())
        self.assertIn("clean digital rendering", guidance["negative_guidance"])
        self.assertIn("flat lighting", guidance["negative_guidance"])
        self.assertTrue(any("desolation" in note.lower() for note in guidance["conditioning_notes"]))


if __name__ == "__main__":
    unittest.main()
