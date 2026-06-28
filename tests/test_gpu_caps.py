from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class ClassifyCapabilitiesTests(unittest.TestCase):
    def test_rtx_3090_ampere_has_bf16_but_no_fp8_compute(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=8, minor=6, name="NVIDIA GeForce RTX 3090", vram_total_gb=24.0)
        self.assertEqual(caps["arch"], "Ampere")
        self.assertTrue(caps["supports_bf16"])
        self.assertFalse(caps["supports_fp8_compute"])
        self.assertFalse(caps["supports_nvfp4"])
        # fp8 still usable as storage-only (saves VRAM, upcast at compute)
        self.assertTrue(caps["fp8_storage_only"])

    def test_rtx_4090_ada_has_fp8_compute(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=8, minor=9, name="NVIDIA GeForce RTX 4090", vram_total_gb=24.0)
        self.assertEqual(caps["arch"], "Ada Lovelace")
        self.assertTrue(caps["supports_fp8_compute"])
        self.assertFalse(caps["fp8_storage_only"])
        self.assertFalse(caps["supports_nvfp4"])

    def test_rtx_5090_blackwell_has_fp8_and_nvfp4(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=12, minor=0, name="NVIDIA GeForce RTX 5090", vram_total_gb=32.0)
        self.assertEqual(caps["arch"], "Blackwell")
        self.assertTrue(caps["supports_fp8_compute"])
        self.assertTrue(caps["supports_nvfp4"])

    def test_hopper_has_fp8_compute(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=9, minor=0, name="NVIDIA H100", vram_total_gb=80.0)
        self.assertTrue(caps["supports_fp8_compute"])

    def test_pascal_is_legacy_no_bf16(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=6, minor=1, name="NVIDIA GTX 1080 Ti", vram_total_gb=11.0)
        self.assertFalse(caps["supports_bf16"])
        self.assertFalse(caps["supports_fp8_compute"])


class ComputeDtypeTests(unittest.TestCase):
    def test_ampere_plus_uses_bf16(self) -> None:
        from gpu_caps import classify_capabilities

        for major, minor in [(8, 6), (8, 9), (9, 0), (12, 0)]:
            caps = classify_capabilities(major=major, minor=minor, name="x", vram_total_gb=24.0)
            self.assertEqual(caps["recommended_compute_dtype"], "bf16")

    def test_turing_volta_use_fp16_not_bf16(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=7, minor=5, name="NVIDIA GeForce RTX 2070", vram_total_gb=8.0)
        self.assertFalse(caps["supports_bf16"])
        self.assertTrue(caps["supports_fp16"])
        self.assertEqual(caps["recommended_compute_dtype"], "fp16")

    def test_pre_pascal_is_unsupported(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=5, minor=0, name="GTX 980", vram_total_gb=4.0)
        self.assertFalse(caps["supports_fp16"])
        self.assertEqual(caps["recommended_compute_dtype"], "fp32")


class RunnabilityTests(unittest.TestCase):
    def test_4090_is_full_tier(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=8, minor=9, name="RTX 4090", vram_total_gb=24.0)
        v = assess_runnability(caps, ram_total_gb=64.0)
        self.assertTrue(v["can_run"])
        self.assertEqual(v["tier"], "high")

    def test_3060_12gb_runs_comfortably(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=8, minor=6, name="RTX 3060", vram_total_gb=12.0)
        v = assess_runnability(caps, ram_total_gb=32.0)
        self.assertTrue(v["can_run"])
        self.assertIn(v["tier"], ("comfortable", "minimum"))
        self.assertEqual(v["compute_dtype"], "bf16")

    def test_2070_8gb_runs_minimum_with_fp16_and_heavy_swap(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=7, minor=5, name="RTX 2070", vram_total_gb=8.0)
        v = assess_runnability(caps, ram_total_gb=32.0)
        self.assertTrue(v["can_run"])
        self.assertEqual(v["tier"], "minimum")
        self.assertEqual(v["compute_dtype"], "fp16")
        self.assertGreater(v["blocks_to_swap"], 0)

    def test_low_vram_blocked(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=8, minor=6, name="RTX 3050 4GB", vram_total_gb=4.0)
        v = assess_runnability(caps, ram_total_gb=16.0)
        self.assertFalse(v["can_run"])
        self.assertIn("vram", v["reason"].lower())

    def test_low_ram_blocked_even_with_decent_gpu(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=8, minor=6, name="RTX 3060", vram_total_gb=12.0)
        v = assess_runnability(caps, ram_total_gb=8.0)
        self.assertFalse(v["can_run"])
        self.assertIn("ram", v["reason"].lower())

    def test_no_cuda_blocked(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=None, minor=None, name="cpu", vram_total_gb=None)
        v = assess_runnability(caps, ram_total_gb=32.0)
        self.assertFalse(v["can_run"])

    def test_old_arch_blocked(self) -> None:
        from gpu_caps import assess_runnability, classify_capabilities

        caps = classify_capabilities(major=5, minor=2, name="GTX 980", vram_total_gb=8.0)
        v = assess_runnability(caps, ram_total_gb=32.0)
        self.assertFalse(v["can_run"])


class CapabilitySummaryTests(unittest.TestCase):
    def test_fp8_note_explains_storage_only_on_ampere(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=8, minor=6, name="RTX 3090", vram_total_gb=24.0)
        self.assertIn("storage-only", caps["fp8_note"].lower())

    def test_unknown_caps_are_conservative(self) -> None:
        from gpu_caps import classify_capabilities

        caps = classify_capabilities(major=None, minor=None, name="", vram_total_gb=None)
        self.assertFalse(caps["supports_fp8_compute"])
        self.assertEqual(caps["arch"], "unknown")


if __name__ == "__main__":
    unittest.main()
