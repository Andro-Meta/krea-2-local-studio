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

from krea2.sampling import _differential_mask_for_timestep


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


if __name__ == "__main__":
    unittest.main()
