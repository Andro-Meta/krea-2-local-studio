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


class LanPaintSamplerTests(unittest.TestCase):
    def test_masked_inner_update_preserves_known_region_and_calls_model(self) -> None:
        from krea2.lanpaint_sampler import lanpaint_masked_inner_update

        calls: list[float] = []
        current = torch.zeros(1, 4, 2)
        known = torch.ones(1, 4, 2)
        mask = torch.tensor([[[0.0], [1.0], [0.0], [1.0]]])

        def velocity_fn(latent: torch.Tensor, tcurr: float) -> torch.Tensor:
            calls.append(float(tcurr))
            return torch.ones_like(latent) * 0.5

        updated = lanpaint_masked_inner_update(
            current,
            known,
            mask,
            tcurr=0.8,
            tprev=0.6,
            velocity_fn=velocity_fn,
            inner_steps=3,
            strength=1.0,
        )

        self.assertEqual(len(calls), 3)
        self.assertTrue(torch.allclose(updated[:, [0, 2]], known[:, [0, 2]]))
        self.assertFalse(torch.allclose(updated[:, [1, 3]], current[:, [1, 3]]))


if __name__ == "__main__":
    unittest.main()
