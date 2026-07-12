from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from moodboard_enrichment import (  # noqa: E402
    GUIDANCE_VERSION,
    MoodboardSource,
    find_subject_violations,
    generate_moodboard_guidance,
    parse_style_schema_json,
    sanitize_transferable_guidance,
)
import moodboard_enrichment  # noqa: E402
from comfy_client import ComfyExecutionError  # noqa: E402


class MoodboardEnrichmentTests(unittest.TestCase):
    def test_cancelled_comfy_guidance_never_falls_back_to_transformers(self) -> None:
        with (
            patch.object(
                moodboard_enrichment,
                "_comfy_qwen_generate",
                side_effect=ComfyExecutionError("ComfyUI execution was interrupted."),
            ),
            patch.object(moodboard_enrichment, "_local_qwen_generate") as fallback,
            patch("settings.settings.local_llm_backend", "comfy"),
        ):
            with self.assertRaisesRegex(ComfyExecutionError, "interrupted"):
                moodboard_enrichment._qwen_generate(
                    "prompt",
                    [],
                    cancel_probe=lambda: True,
                )
        fallback.assert_not_called()

    def test_parses_legacy_guidance_from_fenced_json(self) -> None:
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

        guidance = parse_style_schema_json(text)

        self.assertEqual(guidance["guidance_version"], GUIDANCE_VERSION)
        self.assertIn("low-key", guidance["prompt_guidance"])
        self.assertEqual(guidance["style_axes"], ["gritty realism", "shallow depth of field"])

    def test_parses_style_schema_and_assembles_prompt_guidance(self) -> None:
        text = """
        {
          "palette": "Deep indigo and cyan with warm sodium accents",
          "lighting": "Low-key volumetric light with hard rim contrast",
          "medium_texture": "Heavy 35mm film grain and soft halation",
          "composition": "Center-weighted framing with generous negative space",
          "atmosphere": "Melancholic, hushed, nocturnal",
          "era_or_movement": "1980s neo-noir cinema",
          "style_axes": ["indigo palette", "volumetric light", "film grain"],
          "negative_style_terms": ["flat lighting", "oversaturated colors", "avoid sharp detail", "photorealistic rendering", "crowded scenes"],
          "source_summary": "A nocturnal neo-noir rendering treatment."
        }
        """

        guidance = parse_style_schema_json(text)

        self.assertEqual(guidance["guidance_version"], GUIDANCE_VERSION)
        self.assertIn("Palette: Deep indigo and cyan", guidance["prompt_guidance"])
        self.assertIn("Lighting: Low-key volumetric light", guidance["prompt_guidance"])
        self.assertIn("Era or movement: 1980s neo-noir cinema", guidance["prompt_guidance"])
        # Quality bans and subject bans are filtered from negatives.
        self.assertIn("flat lighting", guidance["negative_guidance"])
        self.assertIn("oversaturated colors", guidance["negative_guidance"])
        self.assertNotIn("sharp detail", guidance["negative_guidance"])
        self.assertNotIn("photorealistic", guidance["negative_guidance"])
        self.assertNotIn("crowded", guidance["negative_guidance"])

    def test_retry_loop_regenerates_when_subjects_leak(self) -> None:
        leaking = """
        {
          "palette": "Slate grey and amber",
          "lighting": "A lone figure backlit by fog lamps",
          "medium_texture": "Film grain",
          "composition": "Centered",
          "atmosphere": "Quiet",
          "era_or_movement": "",
          "style_axes": ["fog"],
          "negative_style_terms": ["flat lighting"],
          "source_summary": "Foggy noir treatment."
        }
        """
        clean = leaking.replace("A lone figure backlit by fog lamps", "Backlit fog glow with a single hard contrast point")
        responses = [leaking, clean]
        prompts: list[str] = []

        def generator(prompt: str, _images: list[str]) -> str:
            prompts.append(prompt)
            return responses[min(len(prompts) - 1, len(responses) - 1)]

        guidance = generate_moodboard_guidance(
            [MoodboardSource(title="Fog Noir", taste_profile="Fog.", keywords=["fog"])],
            mode="official",
            generator=generator,
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("leaked subject matter", prompts[1])
        self.assertIn("figure", prompts[1].lower())
        self.assertNotIn("lone figure", guidance["prompt_guidance"].lower())
        self.assertEqual(find_subject_violations(guidance), [])

    def test_retry_exhaustion_falls_back_to_regex_sanitization(self) -> None:
        leaking = """
        {
          "palette": "Slate grey",
          "lighting": "A lone figure under a streetlamp",
          "medium_texture": "Film grain",
          "composition": "Centered",
          "atmosphere": "Quiet",
          "era_or_movement": "",
          "style_axes": ["solitary silhouette", "film grain"],
          "negative_style_terms": ["flat lighting"],
          "source_summary": "Noir treatment."
        }
        """
        calls: list[int] = []

        def generator(_prompt: str, _images: list[str]) -> str:
            calls.append(1)
            return leaking

        guidance = generate_moodboard_guidance(
            [MoodboardSource(title="Fog Noir", taste_profile="Fog.", keywords=["fog"])],
            mode="official",
            generator=generator,
        )

        self.assertEqual(len(calls), 3)
        blob = " ".join([guidance["prompt_guidance"], " ".join(guidance["style_axes"])]).lower()
        self.assertNotIn("lone figure", blob)
        self.assertNotIn("solitary silhouette", blob)
        self.assertIn("film grain", blob)

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

    def test_sanitizes_literal_source_subject_examples(self) -> None:
        guidance = sanitize_transferable_guidance({
            "prompt_guidance": (
                "A black and white portrait of a young woman with short hair, heavily distorted by digital glitch effects. "
                "Aerial top-down view of a single kayaker in a white kayak on deep blue ocean water. "
                "A single solitary figure stands on a hilltop, dwarfed by the immense scale of the terrain. "
                "Use chromatic aberration, halftone print texture, and moody lighting."
            ),
            "negative_guidance": "Avoid human faces, crowds, text, buildings. Avoid flat lighting.",
            "style_axes": ["Solitary figure", "Aerial human subject", "halftone print"],
            "conditioning_notes": [],
            "source_summary": "A source image with a young woman, a kayaker, and a solitary figure.",
            "guidance_version": 1,
        })

        text = " ".join([
            guidance["prompt_guidance"],
            guidance["negative_guidance"],
            " ".join(guidance["style_axes"]),
        ]).lower()
        for forbidden in ("young woman", "single kayaker", "solitary figure", "human faces", "crowds", "text", "buildings"):
            self.assertIsNone(re.search(rf"\b{re.escape(forbidden)}\b", text), forbidden)
        self.assertIn("user-requested subject", text)
        self.assertIn("chromatic aberration", guidance["prompt_guidance"])
        self.assertIn("halftone", guidance["prompt_guidance"])


if __name__ == "__main__":
    unittest.main()
