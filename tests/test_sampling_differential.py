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

from krea2.sampling import _differential_mask_for_timestep, sample


class DifferentialMaskTests(unittest.TestCase):
    def test_threshold_reveals_gray_mask_values_over_time(self) -> None:
        mask = torch.tensor([[[0.0], [0.25], [0.5], [0.75], [1.0]]])

        early = _differential_mask_for_timestep(mask, tcurr=0.75, t_start=1.0)
        late = _differential_mask_for_timestep(mask, tcurr=0.25, t_start=1.0)

        self.assertTrue(torch.equal(early[0, :, 0] > 0, torch.tensor([False, False, False, True, True])))
        self.assertTrue(torch.equal(late[0, :, 0] > 0, torch.tensor([False, True, True, True, True])))

    def test_strength_blends_binary_schedule_with_original_mask(self) -> None:
        mask = torch.tensor([[[0.25], [0.75]]])

        active = _differential_mask_for_timestep(mask, tcurr=0.5, t_start=1.0, strength=0.5)

        self.assertTrue(torch.allclose(active[0, :, 0], torch.tensor([0.125, 0.875])))

    def test_unknown_sampler_name_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown sampler"):
            sample(
                model=torch.nn.Identity(),
                ae=None,
                encoder=None,
                prompts=None,
                txt=torch.zeros(1, 1, 4),
                txtmask=torch.ones(1, 1, dtype=torch.bool),
                device="cpu",
                dtype=torch.float32,
                width=16,
                height=16,
                steps=1,
                sampler="missing_sampler",
            )

    def test_seed_variance_off_is_bitwise_equivalent(self) -> None:
        from seed_variance import apply_seed_variance

        txt = torch.randn(1, 8, 2, 4)

        out = apply_seed_variance(txt, seed=123, preset="off")

        self.assertTrue(torch.equal(out, txt))

    def test_seed_variance_is_deterministic_and_changes_conditioning(self) -> None:
        from seed_variance import apply_seed_variance

        txt = torch.zeros(1, 8, 2, 4)

        first = apply_seed_variance(txt, seed=123, preset="balanced", protection="none")
        second = apply_seed_variance(txt, seed=123, preset="balanced", protection="none")
        other = apply_seed_variance(txt, seed=124, preset="balanced", protection="none")

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, txt))
        self.assertFalse(torch.equal(first, other))

    def test_seed_variance_protects_first_half_tokens(self) -> None:
        from seed_variance import apply_seed_variance

        txt = torch.zeros(1, 8, 2, 4)

        out = apply_seed_variance(txt, seed=123, preset="bold", protection="first_half")

        self.assertTrue(torch.equal(out[:, :4], txt[:, :4]))
        self.assertFalse(torch.equal(out[:, 4:], txt[:, 4:]))


if __name__ == "__main__":
    unittest.main()
