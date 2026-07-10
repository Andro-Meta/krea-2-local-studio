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

    def test_generation_request_accepts_image_prompt_modes(self) -> None:
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="x")
        self.assertFalse(req.image_prompt_enabled)
        self.assertEqual(req.image_prompt_mode, "match_style")
        self.assertEqual(req.image_prompt_strength, 0.2)

        for mode in ("match_style", "copy_composition"):
            with self.subTest(mode=mode):
                self.assertEqual(GenerationRequest(prompt="x", image_prompt_mode=mode).image_prompt_mode, mode)

        with self.assertRaises(ValidationError):
            GenerationRequest(prompt="x", image_prompt_mode="invalid")

    def test_image_prompt_strength_rejects_node_invalid_values(self) -> None:
        from schemas import GenerationRequest

        for value in (0.1, 0.2, 1.0):
            with self.subTest(value=value):
                self.assertEqual(GenerationRequest(prompt="x", image_prompt_strength=value).image_prompt_strength, value)

        for value in (0.05, 1.05):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    GenerationRequest(prompt="x", image_prompt_strength=value)

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

    def test_match_style_mode_averages_separate_ref_encodes(self) -> None:
        import comfy_workflows
        from comfy_workflows import GraphBuilder, _build_positive
        from schemas import GenerationRequest, StyleReferenceInput

        req = GenerationRequest(
            prompt="a quiet forest",
            image_prompt_enabled=True,
            image_prompt_mode="match_style",
            image_prompt_strength=0.2,
            style_references=[
                StyleReferenceInput(image_b64="a"),
                StyleReferenceInput(image_b64="b"),
                StyleReferenceInput(image_b64="c"),
            ],
            seed_variance_preset="off",
        )
        g = GraphBuilder()

        original = comfy_workflows._b64_to_loadimage
        comfy_workflows._b64_to_loadimage = lambda graph, b64: graph.add("LoadImage", {"image": f"{b64}.png"})
        try:
            _build_positive(g, req, ["clip", 0], seed=123)
        finally:
            comfy_workflows._b64_to_loadimage = original
        graph = g.graph()

        self.assertEqual(
            sum(1 for node in graph.values() if node["class_type"] == "TextEncodeKrea2"),
            3,
        )
        self.assertGreaterEqual(
            sum(1 for node in graph.values() if node["class_type"] == "ConditioningAverage"),
            2,
        )
        self.assertFalse(any(node["class_type"] == "Krea2EncodeRebalance" for node in graph.values()))

    def test_copy_composition_mode_uses_multi_image_encoder(self) -> None:
        import comfy_workflows
        from comfy_workflows import GraphBuilder, _build_positive
        from schemas import GenerationRequest, StyleReferenceInput

        req = GenerationRequest(
            prompt="a quiet forest",
            image_prompt_enabled=True,
            image_prompt_mode="copy_composition",
            style_references=[
                StyleReferenceInput(image_b64="a"),
                StyleReferenceInput(image_b64="b"),
            ],
            seed_variance_preset="off",
        )
        g = GraphBuilder()

        original = comfy_workflows._b64_to_loadimage
        comfy_workflows._b64_to_loadimage = lambda graph, b64: graph.add("LoadImage", {"image": f"{b64}.png"})
        try:
            _build_positive(g, req, ["clip", 0], seed=123)
        finally:
            comfy_workflows._b64_to_loadimage = original

        self.assertTrue(any(node["class_type"] == "Krea2EncodeRebalance" for node in g.graph().values()))


if __name__ == "__main__":
    unittest.main()
