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


class ConditioningRebalanceTests(unittest.TestCase):
    def test_scale_conditioning_applies_4d_layer_weights(self) -> None:
        from conditioning import scale_conditioning

        txt = torch.ones(1, 2, 12, 2560, dtype=torch.bfloat16)
        out = scale_conditioning(txt, multiplier=2.0, layer_weights=[1.0] * 11 + [3.0])

        self.assertEqual(out.shape, txt.shape)
        self.assertEqual(out.dtype, txt.dtype)
        self.assertEqual(float(out[0, 0, 0, 0]), 2.0)
        self.assertEqual(float(out[0, 0, 11, 0]), 6.0)

    def test_guidance_conditioning_preserves_shape_and_dtype(self) -> None:
        from conditioning import guidance_conditioning

        base = torch.ones(1, 3, 12, 8, dtype=torch.float16)
        guide = torch.full_like(base, 2.0)

        out = guidance_conditioning(base, guide, scale=0.5)

        self.assertEqual(out.shape, base.shape)
        self.assertEqual(out.dtype, base.dtype)

    def test_split_conditioning_schedule_weights(self) -> None:
        from conditioning import split_schedule_weights

        self.assertEqual(split_schedule_weights(0.2), (1.0, 0.0))
        self.assertEqual(split_schedule_weights(0.5), (0.5, 0.5))
        self.assertEqual(split_schedule_weights(0.9), (0.0, 1.0))

    def test_invalid_weights_fall_back_safely(self) -> None:
        from conditioning import DEFAULT_LAYER_WEIGHTS, parse_weights

        self.assertEqual(parse_weights("not,a,weight"), DEFAULT_LAYER_WEIGHTS)
        self.assertEqual(parse_weights("1,2,3"), DEFAULT_LAYER_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
