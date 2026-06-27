from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class PromptPlannerTests(unittest.TestCase):
    def test_parse_planner_json_extracts_prompt_and_fields(self) -> None:
        from prompt_planner import parse_planner_response

        parsed = parse_planner_response(
            """
            ```json
            {
              "planned_prompt": "a red fox in a misty forest",
              "negative_prompt": "blurry, extra limbs",
              "subject": "red fox",
              "composition": "centered portrait",
              "style": "cinematic wildlife photo",
              "lighting": "soft foggy backlight",
              "materials": "wet moss, fur",
              "text_rendering": "",
              "regions": [{"label": "background", "prompt": "misty pine forest"}]
            }
            ```
            """
        )

        self.assertEqual(parsed["planned_prompt"], "a red fox in a misty forest")
        self.assertEqual(parsed["negative_prompt"], "blurry, extra limbs")
        self.assertEqual(parsed["subject"], "red fox")
        self.assertEqual(parsed["regions"][0]["label"], "background")

    def test_heuristic_planner_preserves_prompt_and_adds_adherence_structure(self) -> None:
        from prompt_planner import plan_prompt_heuristic

        result = plan_prompt_heuristic("silver robot holding a sign that says \"KREA\"")

        self.assertIn("silver robot", result.planned_prompt)
        self.assertIn('"KREA"', result.planned_prompt)
        self.assertTrue(result.changed)
        self.assertEqual(result.backend, "heuristic")

    def test_generation_metadata_records_prompt_plan(self) -> None:
        from generation_metadata import build_generation_metadata

        req = SimpleNamespace(
            prompt="expanded prompt",
            negative_prompt="",
            mode="txt2img",
            model_profile="",
            checkpoint="turbo",
            checkpoint_path="",
            quantization="fp8",
            steps=8,
            cfg=0.0,
            width=1024,
            height=1024,
            denoise=1.0,
            sampler="euler_flow",
            scheduler="simple",
            inpaint_method="native",
            lanpaint_inner_steps=3,
            lanpaint_strength=1.0,
            lanpaint_lambda=16.0,
            lanpaint_step_size=0.2,
            lanpaint_beta=1.0,
            lanpaint_friction=15.0,
            lanpaint_early_stop=1,
            lanpaint_prompt_mode="Image First",
            mu=1.15,
            y1=0.5,
            y2=1.15,
            edit_provider="auto",
            quality_preset="balanced",
            creativity="medium",
            conditioning_mode="auto",
            loras=[],
            mood="",
            moodboard_ids=[],
            moodboard_uuids=[],
            moodboard_strength=0.35,
            moodboard_images=[],
            style_references=[],
            style_fusion_mode="semantic_fusion",
            seed_variance_preset="off",
            seed_variance_strength=0.0,
            seed_variance_protection="first_half",
            use_rebalance=True,
            rebalance_mode="rms_renorm",
            rebalance_preset="balanced",
            rebalance_renormalize=True,
            rebalance_multiplier=1.0,
            rebalance_weights="",
            edit_rebalance_enabled=True,
            edit_rebalance_profile="conservative",
            krea_enhancer_enabled=False,
            krea_enhancer_variant="off",
            krea_enhancer_strength=1.0,
            krea_enhancer_delta_cap=0.75,
            refine=False,
            refine_denoise=0.3,
            refine_steps=6,
            bboxes=[],
            use_prompt_expander=False,
            use_prompt_planner=True,
            prompt_planner_max_tokens=700,
            prompt_planner_output={
                "original_prompt": "rough prompt",
                "planned_prompt": "expanded prompt",
                "backend": "heuristic",
            },
        )

        metadata = build_generation_metadata(req, base_seed=12)

        self.assertTrue(metadata["prompt_planner"]["enabled"])
        self.assertEqual(metadata["prompt_planner"]["output"]["original_prompt"], "rough prompt")


if __name__ == "__main__":
    unittest.main()
