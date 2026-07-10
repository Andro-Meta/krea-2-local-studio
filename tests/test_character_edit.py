from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class CharacterEditGraphTests(unittest.TestCase):
    def test_character_edit_graph_uses_ostris_nodes_and_identity_lora(self) -> None:
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="restage this person at a night market",
            mode="character_edit",
            character_edit_source_b64="iVBORw0KGgo=",
            checkpoint="turbo",
            quantization="int8",
            diffusion_engine="native_int8_convrot",
            steps=8,
            cfg=1.0,
        )

        with patch("comfy_workflows._b64_to_loadimage", return_value="source_image"):
            graph, _runtime = build_graph(req)
        classes = {node["class_type"] for node in graph.values()}

        self.assertIn("Krea2EditGroundedEncode", classes)
        self.assertIn("Krea2EditModelPatch", classes)
        grounded = [node for node in graph.values() if node["class_type"] == "Krea2EditGroundedEncode"]
        self.assertEqual(grounded[0]["inputs"]["grounding_px"], 768)
        identity_nodes = [
            node for node in graph.values()
            if node["class_type"] == "LoraLoaderModelOnly"
            and node["inputs"].get("lora_name") == "krea2_identity_edit_v1.safetensors"
        ]
        self.assertEqual(len(identity_nodes), 1)

    def test_character_edit_grounds_empty_negative_at_high_cfg(self) -> None:
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="remove the object",
            negative_prompt="blurry, deformed",
            mode="character_edit",
            character_edit_source_b64="iVBORw0KGgo=",
            checkpoint="raw",
            steps=20,
            cfg=3.0,
        )

        with patch("comfy_workflows._b64_to_loadimage", return_value="source_image"):
            graph, _runtime = build_graph(req)

        grounded = [node for node in graph.values() if node["class_type"] == "Krea2EditGroundedEncode"]
        # Two grounded encoders (positive + negative); the negative must be empty.
        self.assertEqual(len(grounded), 2)
        prompts = sorted(node["inputs"]["prompt"] for node in grounded)
        self.assertEqual(prompts[0], "")
        self.assertNotIn("blurry, deformed", prompts)

    def test_character_edit_two_reference_wires_scene_as_primary(self) -> None:
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="place this person in the scene",
            mode="character_edit",
            character_edit_source_b64="c3ViamVjdA==",
            character_edit_reference_b64="c2NlbmU=",
            checkpoint="turbo",
            steps=8,
            cfg=1.0,
        )

        def fake_load(_g, b64):
            return f"img_{b64}"

        with patch("comfy_workflows._b64_to_loadimage", side_effect=fake_load):
            graph, _runtime = build_graph(req)

        patches = [n for n in graph.values() if n["class_type"] == "Krea2EditModelPatch"]
        self.assertEqual(len(patches), 1)
        self.assertIn("source_latent_b", patches[0]["inputs"])
        grounded = [n for n in graph.values() if n["class_type"] == "Krea2EditGroundedEncode"]
        self.assertTrue(all("image_b" in n["inputs"] for n in grounded))

    def test_character_edit_single_reference_has_no_b_inputs(self) -> None:
        from comfy_workflows import build_graph
        from schemas import GenerationRequest

        req = GenerationRequest(
            prompt="edit",
            mode="character_edit",
            character_edit_source_b64="c3ViamVjdA==",
            checkpoint="turbo",
            steps=8,
            cfg=1.0,
        )
        with patch("comfy_workflows._b64_to_loadimage", return_value="source_image"):
            graph, _runtime = build_graph(req)

        patch_node = [n for n in graph.values() if n["class_type"] == "Krea2EditModelPatch"][0]
        self.assertNotIn("source_latent_b", patch_node["inputs"])
        grounded = [n for n in graph.values() if n["class_type"] == "Krea2EditGroundedEncode"]
        self.assertTrue(all("image_b" not in n["inputs"] for n in grounded))

    def test_character_edit_regions_apply_masked_conditioning(self) -> None:
        from comfy_workflows import build_graph
        from schemas import CharacterEditRegion, GenerationRequest

        req = GenerationRequest(
            prompt="two people at a cafe",
            mode="character_edit",
            character_edit_source_b64="c3ViamVjdA==",
            checkpoint="turbo",
            steps=8,
            cfg=1.0,
            character_edit_regions=[
                CharacterEditRegion(x=0.0, y=0.0, w=0.5, h=1.0, prompt="person A on the left", reference_b64="cGVyc29uQQ=="),
                CharacterEditRegion(x=0.5, y=0.0, w=0.5, h=1.0, prompt="person B on the right", reference_b64="cGVyc29uQg=="),
            ],
        )

        with patch("comfy_workflows._b64_to_loadimage", return_value="img"):
            graph, _runtime = build_graph(req)

        classes = [n["class_type"] for n in graph.values()]
        self.assertEqual(classes.count("ConditioningSetMask"), 2)
        self.assertGreaterEqual(classes.count("ConditioningCombine"), 2)
        self.assertEqual(classes.count("ImageToMask"), 2)

    def test_character_edit_skips_empty_region_prompts(self) -> None:
        from comfy_workflows import build_graph
        from schemas import CharacterEditRegion, GenerationRequest

        req = GenerationRequest(
            prompt="scene",
            mode="character_edit",
            character_edit_source_b64="c3ViamVjdA==",
            checkpoint="turbo",
            steps=8,
            cfg=1.0,
            character_edit_regions=[CharacterEditRegion(x=0.0, y=0.0, w=0.5, h=1.0, prompt="   ")],
        )
        with patch("comfy_workflows._b64_to_loadimage", return_value="img"):
            graph, _runtime = build_graph(req)
        classes = [n["class_type"] for n in graph.values()]
        self.assertNotIn("ConditioningSetMask", classes)

    def test_character_edit_request_defaults(self) -> None:
        from schemas import GenerationRequest

        req = GenerationRequest(prompt="x", mode="character_edit")

        self.assertEqual(req.character_edit_grounding_px, 768)
        self.assertEqual(req.character_edit_task, "restage")
        self.assertEqual(req.character_edit_lora_strength, 1.0)

    def test_character_edit_prefers_stock_krea_text_encoder_when_present(self) -> None:
        import comfy_workflows

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            encoder = root / "krea2" / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors"
            encoder.parent.mkdir(parents=True)
            encoder.write_bytes(b"placeholder")
            graph = comfy_workflows.GraphBuilder()
            with patch("settings.MODELS_DIR", root):
                clip = comfy_workflows._build_character_edit_clip(graph)

        node = graph.graph()[clip[0]]
        self.assertEqual(node["inputs"]["clip_name"], "qwen3vl_4b_fp8_scaled.safetensors")


if __name__ == "__main__":
    unittest.main()
