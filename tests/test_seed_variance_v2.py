from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class SeedVarianceV2Tests(unittest.TestCase):
    def test_direction_patterns_shape_noise(self) -> None:
        from seed_variance import apply_seed_variance

        base = torch.zeros((1, 12, 4), dtype=torch.float32)
        forward = apply_seed_variance(base, seed=7, preset="custom", strength=0.05, protection="none", direction="forward")
        reverse = apply_seed_variance(base, seed=7, preset="custom", strength=0.05, protection="none", direction="reverse")

        self.assertGreater(forward[:, -1].abs().mean().item(), forward[:, 0].abs().mean().item())
        self.assertGreater(reverse[:, 0].abs().mean().item(), reverse[:, -1].abs().mean().item())

    def test_timing_window_limits_injection(self) -> None:
        from seed_variance import apply_seed_variance

        base = torch.zeros((1, 20, 3), dtype=torch.float32)
        varied = apply_seed_variance(
            base,
            seed=9,
            preset="custom",
            strength=0.05,
            protection="none",
            injection_start=0.25,
            injection_end=0.5,
        )

        self.assertTrue(torch.allclose(varied[:, :5], base[:, :5]))
        self.assertGreater(varied[:, 5:10].abs().sum().item(), 0)
        self.assertTrue(torch.allclose(varied[:, 10:], base[:, 10:]))

    def test_neutral_advanced_controls_match_existing_behavior(self) -> None:
        from seed_variance import apply_seed_variance

        base = torch.zeros((1, 8, 2), dtype=torch.float32)
        old = apply_seed_variance(base, seed=3, preset="balanced", protection="first_half")
        new = apply_seed_variance(
            base,
            seed=3,
            preset="balanced",
            protection="first_half",
            direction="none",
            fade_curve="linear",
            injection_start=0.0,
            injection_end=1.0,
        )

        self.assertTrue(torch.allclose(old, new))

    def test_rbg_sparse_mode_changes_only_a_small_fraction(self) -> None:
        from seed_variance import apply_seed_variance

        base = torch.zeros((1, 20, 100), dtype=torch.float32)
        varied = apply_seed_variance(
            base,
            seed=11,
            preset="creative",
            protection="none",
            algorithm="rbg",
            model_type="krea2",
            direction="facevar",
            shift_strength=170,
            randomize_percent=None,
            fade_curve="instant",
        )

        changed = (varied != base).sum().item()
        self.assertGreater(changed, 0)
        self.assertLess(changed, base.numel() * 0.08)

    def test_rbg_mode_supports_last_half_protection(self) -> None:
        from seed_variance import apply_seed_variance

        base = torch.zeros((1, 12, 16), dtype=torch.float32)
        varied = apply_seed_variance(
            base,
            seed=12,
            preset="bold",
            protection="last_half",
            algorithm="rbg",
            direction="realistic",
            fade_curve="instant",
        )

        self.assertGreater(varied[:, :6].abs().sum().item(), 0)
        self.assertTrue(torch.equal(varied[:, 6:], base[:, 6:]))

    def test_rbg_quality_default_matches_expression_recipe(self) -> None:
        from seed_variance import rbg_quality_defaults

        defaults = rbg_quality_defaults()

        self.assertEqual(defaults["algorithm"], "rbg")
        self.assertEqual(defaults["preset"], "creative")
        self.assertEqual(defaults["model_type"], "krea2")
        self.assertEqual(defaults["direction"], "visceral_expression_grit")
        self.assertEqual(defaults["shift_strength"], 170)
        self.assertEqual(defaults["schedule"], "step_cutoff")
        self.assertEqual(defaults["cutoff_step"], 3)
        self.assertEqual(defaults["total_steps"], 13)
        self.assertAlmostEqual(defaults["cutoff_strength"], 0.53)


if __name__ == "__main__":
    unittest.main()
