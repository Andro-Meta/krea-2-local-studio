from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class RecommendRuntimeTests(unittest.TestCase):
    def test_fp8_1k_fits_without_swapping(self) -> None:
        from resource_manager import recommend_runtime

        rec = recommend_runtime(free_vram_gb=24.0, width=1024, height=1024, quantization="fp8")
        self.assertTrue(rec["fits"])
        self.assertEqual(rec["blocks_to_swap"], 0)
        self.assertFalse(rec["tiled_decode"])

    def test_bf16_2k_on_tight_vram_recommends_block_swap(self) -> None:
        from resource_manager import recommend_runtime

        rec = recommend_runtime(free_vram_gb=12.0, width=2048, height=2048, quantization="bf16")
        self.assertGreater(rec["blocks_to_swap"], 0)
        self.assertTrue(rec["warnings"])  # heavy 2k warning

    def test_2k_flags_tiled_decode(self) -> None:
        from resource_manager import recommend_runtime

        rec = recommend_runtime(free_vram_gb=24.0, width=2048, height=2048, quantization="fp8")
        self.assertTrue(rec["tiled_decode"])

    def test_unknown_vram_is_advisory_only(self) -> None:
        from resource_manager import recommend_runtime

        rec = recommend_runtime(free_vram_gb=None, width=1024, height=1024, quantization="fp8")
        self.assertTrue(rec["fits"])
        self.assertEqual(rec["blocks_to_swap"], 0)

    def test_blocks_never_exceed_total(self) -> None:
        from resource_manager import recommend_runtime

        rec = recommend_runtime(free_vram_gb=1.0, width=2048, height=2048, quantization="bf16", total_blocks=28)
        self.assertLessEqual(rec["blocks_to_swap"], 28)


class EstimateTests(unittest.TestCase):
    def test_inference_scratch_scales_with_area_and_cfg(self) -> None:
        from resource_manager import estimate_inference_scratch_gb

        one_k = estimate_inference_scratch_gb(1024, 1024, batch=1, cfg_active=False)
        two_k = estimate_inference_scratch_gb(2048, 2048, batch=1, cfg_active=False)
        one_k_cfg = estimate_inference_scratch_gb(1024, 1024, batch=1, cfg_active=True)

        # 2K is ~4x the latent area of 1K.
        self.assertAlmostEqual(two_k / one_k, 4.0, delta=0.2)
        # CFG doubles the effective batch (pos+neg).
        self.assertAlmostEqual(one_k_cfg / one_k, 2.0, delta=0.05)


class RecommendDefaultsTests(unittest.TestCase):
    def test_ampere_24gb_prefers_fp8_storage_with_note(self) -> None:
        from gpu_caps import classify_capabilities
        from resource_manager import recommend_defaults

        caps = classify_capabilities(major=8, minor=6, name="RTX 3090", vram_total_gb=24.0)
        rec = recommend_defaults(caps, free_vram_gb=23.0)
        self.assertEqual(rec["quantization"], "fp8")  # storage-only still saves VRAM
        self.assertIn("storage-only", rec["notes"].lower())

    def test_ada_24gb_recommends_fp8(self) -> None:
        from gpu_caps import classify_capabilities
        from resource_manager import recommend_defaults

        caps = classify_capabilities(major=8, minor=9, name="RTX 4090", vram_total_gb=24.0)
        rec = recommend_defaults(caps, free_vram_gb=23.0)
        self.assertEqual(rec["quantization"], "fp8")

    def test_blackwell_32gb_allows_bf16(self) -> None:
        from gpu_caps import classify_capabilities
        from resource_manager import recommend_defaults

        caps = classify_capabilities(major=12, minor=0, name="RTX 5090", vram_total_gb=32.0)
        rec = recommend_defaults(caps, free_vram_gb=31.0)
        self.assertIn(rec["quantization"], ("bf16", "fp8"))
        self.assertEqual(rec["max_tier"], "2k")

    def test_small_vram_caps_to_1k_and_swaps(self) -> None:
        from gpu_caps import classify_capabilities
        from resource_manager import recommend_defaults

        caps = classify_capabilities(major=8, minor=6, name="RTX 3060", vram_total_gb=12.0)
        rec = recommend_defaults(caps, free_vram_gb=11.0)
        self.assertEqual(rec["quantization"], "fp8")
        self.assertGreater(rec["blocks_to_swap"], 0)


class PlanGenerationTests(unittest.TestCase):
    def test_parallel_batch_allows_turbo_1k_when_estimate_fits(self) -> None:
        from resource_manager import plan_parallel_batch

        plan = plan_parallel_batch(
            free_vram_gb=18.0,
            width=1024,
            height=1024,
            quantization="fp8",
            batch=2,
            cfg_active=False,
            mode="txt2img",
            checkpoint="turbo",
        )

        self.assertTrue(plan["allowed"])
        self.assertTrue(plan["fits"])

    def test_parallel_batch_requires_strict_headroom_and_small_batch(self) -> None:
        from resource_manager import plan_parallel_batch

        low_headroom = plan_parallel_batch(
            free_vram_gb=10.7,
            width=1024,
            height=1024,
            quantization="fp8",
            batch=2,
            cfg_active=False,
            mode="txt2img",
            checkpoint="turbo",
        )
        high_batch = plan_parallel_batch(
            free_vram_gb=24.0,
            width=1024,
            height=1024,
            quantization="fp8",
            batch=4,
            cfg_active=False,
            mode="txt2img",
            checkpoint="turbo",
        )

        self.assertFalse(low_headroom["allowed"])
        self.assertFalse(high_batch["allowed"])

    def test_parallel_batch_blocks_raw_and_2k(self) -> None:
        from resource_manager import plan_parallel_batch

        raw = plan_parallel_batch(
            free_vram_gb=24.0,
            width=1024,
            height=1024,
            quantization="fp8",
            batch=2,
            cfg_active=True,
            mode="txt2img",
            checkpoint="raw",
        )
        two_k = plan_parallel_batch(
            free_vram_gb=24.0,
            width=2048,
            height=2048,
            quantization="fp8",
            batch=2,
            cfg_active=False,
            mode="txt2img",
            checkpoint="turbo",
        )

        self.assertFalse(raw["allowed"])
        self.assertFalse(two_k["allowed"])

    def test_parallel_batch_blocks_edit_modes_and_tight_vram(self) -> None:
        from resource_manager import plan_parallel_batch

        edit = plan_parallel_batch(
            free_vram_gb=24.0,
            width=1024,
            height=1024,
            quantization="fp8",
            batch=2,
            cfg_active=False,
            mode="inpaint",
            checkpoint="turbo",
        )
        tight = plan_parallel_batch(
            free_vram_gb=1.0,
            width=1024,
            height=1024,
            quantization="fp8",
            batch=4,
            cfg_active=True,
            mode="txt2img",
            checkpoint="turbo",
        )

        self.assertFalse(edit["allowed"])
        self.assertFalse(tight["allowed"])

    def test_tight_vram_triggers_tiled_decode_and_preclear(self) -> None:
        from resource_manager import plan_generation

        plan = plan_generation(free_vram_gb=10.0, width=2048, height=2048, quantization="bf16")
        self.assertTrue(plan["tiled_decode"])
        self.assertTrue(plan["clear_cache_first"])

    def test_ample_vram_1k_keeps_cache_and_no_tiling(self) -> None:
        from resource_manager import plan_generation

        plan = plan_generation(free_vram_gb=24.0, width=1024, height=1024, quantization="fp8")
        self.assertFalse(plan["tiled_decode"])
        self.assertFalse(plan["clear_cache_first"])

    def test_int8_is_estimated_as_low_vram_quantization(self) -> None:
        from resource_manager import recommend_runtime

        rec = recommend_runtime(free_vram_gb=24.0, width=1024, height=1024, quantization="int8")

        self.assertTrue(rec["fits"])
        self.assertEqual(rec["blocks_to_swap"], 0)
        self.assertLess(rec["estimated_vram_gb"], 20)


class CachePolicyTests(unittest.TestCase):
    def test_large_render_triggers_cache_clear(self) -> None:
        from resource_manager import should_clear_after_render

        self.assertTrue(should_clear_after_render(2048, 2048, free_vram_gb=24.0))

    def test_small_render_with_ample_vram_keeps_cache(self) -> None:
        from resource_manager import should_clear_after_render

        self.assertFalse(should_clear_after_render(1024, 1024, free_vram_gb=24.0))

    def test_low_free_vram_forces_clear_even_when_small(self) -> None:
        from resource_manager import should_clear_after_render

        self.assertTrue(should_clear_after_render(768, 768, free_vram_gb=1.5))


if __name__ == "__main__":
    unittest.main()
