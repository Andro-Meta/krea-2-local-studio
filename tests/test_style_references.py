from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class StyleReferenceSchemaTests(unittest.TestCase):
    def test_style_reference_defaults_match_comfy_node(self) -> None:
        from schemas import StyleReferenceInput

        ref = StyleReferenceInput(image_b64="abc")

        self.assertEqual(ref.strength, 1.0)
        self.assertEqual(ref.role, "style")
        self.assertEqual(ref.token_size, "normal")

    def test_style_reference_roles_are_validated(self) -> None:
        from schemas import StyleReferenceInput

        for role in ("style", "layout", "subject", "mood", "texture", "target"):
            with self.subTest(role=role):
                self.assertEqual(StyleReferenceInput(image_b64="abc", role=role).role, role)

        with self.assertRaises(ValidationError):
            StyleReferenceInput(image_b64="abc", role="invalid")

    def test_generation_request_accepts_style_fusion_modes(self) -> None:
        from schemas import GenerationRequest

        for mode in ("style_only", "preserve_structure", "semantic_fusion"):
            with self.subTest(mode=mode):
                self.assertEqual(GenerationRequest(prompt="x", style_fusion_mode=mode).style_fusion_mode, mode)

        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="x", style_fusion_mode="invalid")

    def test_style_reference_strength_accepts_comfy_range(self) -> None:
        from schemas import StyleReferenceInput

        for strength in (-2.0, 0.0, 1.0, 2.0):
            with self.subTest(strength=strength):
                self.assertEqual(StyleReferenceInput(image_b64="abc", strength=strength).strength, strength)

    def test_style_reference_strength_rejects_out_of_range_values(self) -> None:
        from schemas import StyleReferenceInput

        for strength in (-2.05, 2.05):
            with self.subTest(strength=strength):
                with self.assertRaises(ValidationError):
                    StyleReferenceInput(image_b64="abc", strength=strength)

    def test_generation_request_accepts_at_most_ten_style_references(self) -> None:
        from schemas import GenerationRequest, StyleReferenceInput

        refs = [StyleReferenceInput(image_b64=str(i)) for i in range(10)]
        req = GenerationRequest(prompt="a quiet forest", style_references=refs)

        self.assertEqual(len(req.style_references), 10)

        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="a quiet forest", style_references=refs + [StyleReferenceInput(image_b64="extra")])

    def test_legacy_reference_images_convert_to_style_references(self) -> None:
        from inference import resolve_style_references

        req = SimpleNamespace(
            style_references=[],
            ref_image1_b64="one",
            ref_image2_b64="",
            ref_image3_b64="three",
        )

        refs = resolve_style_references(req)

        self.assertEqual([ref["image_b64"] for ref in refs], ["one", "three"])
        self.assertEqual([ref["strength"] for ref in refs], [1.0, 1.0])
        self.assertEqual([ref["token_size"] for ref in refs], ["normal", "normal"])

    def test_structured_and_legacy_references_are_capped_to_ten(self) -> None:
        from inference import resolve_style_references
        from schemas import StyleReferenceInput

        req = SimpleNamespace(
            style_references=[StyleReferenceInput(image_b64=str(i), strength=0.5) for i in range(9)],
            ref_image1_b64="legacy-one",
            ref_image2_b64="legacy-two",
            ref_image3_b64="legacy-three",
        )

        refs = resolve_style_references(req)

        self.assertEqual(len(refs), 10)
        self.assertEqual(refs[-1]["image_b64"], "legacy-one")

    def test_generation_metadata_round_trips_style_reference_settings(self) -> None:
        from generation_metadata import build_generation_metadata
        from schemas import GenerationRequest, StyleReferenceInput

        req = GenerationRequest(
            prompt="a quiet forest",
            style_references=[
                StyleReferenceInput(image_b64="positive", strength=1.25, token_size="high"),
                StyleReferenceInput(image_b64="negative", strength=-0.5, token_size="low"),
            ],
        )

        metadata = build_generation_metadata(req, base_seed=123)
        refs = metadata["image_references"]["style_references"]

        self.assertEqual(metadata["image_references"]["style_reference_count"], 2)
        self.assertEqual(refs[0]["strength"], 1.25)
        self.assertEqual(refs[0]["token_size"], "high")
        self.assertEqual(refs[1]["strength"], -0.5)
        self.assertEqual(refs[0]["image_b64"], "positive")

    def test_generation_metadata_records_style_fusion_mode(self) -> None:
        from generation_metadata import build_generation_metadata
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="a quiet forest", style_fusion_mode="preserve_structure")

        metadata = build_generation_metadata(req, base_seed=123)

        self.assertEqual(metadata["image_references"]["style_fusion_mode"], "preserve_structure")


if __name__ == "__main__":
    unittest.main()
