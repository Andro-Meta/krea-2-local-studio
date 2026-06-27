from __future__ import annotations

import base64
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def mask_b64(box: tuple[int, int, int, int], size: tuple[int, int] = (32, 32)) -> str:
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=255)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RegionalSceneTests(unittest.TestCase):
    def test_normalize_masks_bounds_overlaps(self) -> None:
        from regional_scene import normalize_region_masks

        masks = normalize_region_masks(
            [
                {"mask_b64": mask_b64((0, 0, 24, 24)), "normalize": True, "feather": 0},
                {"mask_b64": mask_b64((8, 8, 31, 31)), "normalize": True, "feather": 0},
            ],
            size=(32, 32),
        )

        overlap_total = sum(mask.getpixel((12, 12)) for mask in masks)
        self.assertLessEqual(overlap_total, 255)
        self.assertGreater(masks[0].getpixel((2, 2)), 0)

    def test_feather_mask_softens_edge(self) -> None:
        from regional_scene import normalize_region_masks

        masks = normalize_region_masks(
            [{"mask_b64": mask_b64((8, 8, 16, 16)), "normalize": False, "feather": 6}],
            size=(32, 32),
        )

        self.assertGreater(masks[0].getpixel((6, 8)), 0)
        self.assertLess(masks[0].getpixel((6, 8)), 255)

    def test_blend_regional_prompts_keeps_base_and_visible_regions(self) -> None:
        from regional_scene import build_regional_prompt_text

        prompt = build_regional_prompt_text(
            "cinematic alley",
            [
                {"prompt": "left side neon sign", "negative_prompt": "misspelled text", "strength": 0.8, "visible": True},
                {"prompt": "hidden region", "visible": False},
            ],
            base_prompt_strength=0.3,
        )

        self.assertIn("cinematic alley", prompt)
        self.assertIn("left side neon sign", prompt)
        self.assertNotIn("hidden region", prompt)
        self.assertIn("not visible text", prompt)

    def test_metadata_records_regions_without_raw_mask_payloads(self) -> None:
        from generation_metadata import build_generation_metadata

        req = SimpleNamespace(
            prompt="prompt",
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
            regional_prompts=[{"prompt": "neon left", "mask_b64": "abc", "strength": 0.7}],
            regional_base_prompt_strength=0.3,
            regional_normalize_masks=True,
        )

        metadata = build_generation_metadata(req, base_seed=1)

        self.assertEqual(metadata["regional_prompts"]["regions"][0]["prompt"], "neon left")
        self.assertNotIn("mask_b64", metadata["regional_prompts"]["regions"][0])
        self.assertIn("mask_hash", metadata["regional_prompts"]["regions"][0])


if __name__ == "__main__":
    unittest.main()
