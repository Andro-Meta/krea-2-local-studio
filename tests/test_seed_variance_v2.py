from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class SeedVarianceV2Tests(unittest.TestCase):
    def test_direction_patterns_shape_noise(self) -> None:
        import torch
        from seed_variance import apply_seed_variance

        base = torch.zeros((1, 12, 4), dtype=torch.float32)
        forward = apply_seed_variance(base, seed=7, preset="custom", strength=0.05, protection="none", direction="forward")
        reverse = apply_seed_variance(base, seed=7, preset="custom", strength=0.05, protection="none", direction="reverse")

        self.assertGreater(forward[:, -1].abs().mean().item(), forward[:, 0].abs().mean().item())
        self.assertGreater(reverse[:, 0].abs().mean().item(), reverse[:, -1].abs().mean().item())

    def test_timing_window_limits_injection(self) -> None:
        import torch
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
        import torch
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


if __name__ == "__main__":
    unittest.main()
