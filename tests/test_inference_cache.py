from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from inference import Krea2Pipeline


class InferenceCacheTests(unittest.TestCase):
    def test_conditioning_cache_round_trips_cpu_tensors(self) -> None:
        pipeline = Krea2Pipeline()
        txt = torch.ones(1, 2, 3, dtype=torch.bfloat16)
        mask = torch.tensor([[True, False]])

        pipeline._put_conditioning_cache(("positive", "prompt"), txt, mask)
        txt.fill_(5)

        hit = pipeline._get_conditioning_cache(("positive", "prompt"))

        self.assertIsNotNone(hit)
        cached_txt, cached_mask = hit
        self.assertTrue(torch.equal(cached_txt, torch.ones(1, 2, 3, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(cached_mask, mask))
        self.assertEqual(cached_txt.device.type, "cpu")
        self.assertEqual(cached_mask.device.type, "cpu")

    def test_conditioning_cache_evicts_oldest_entry(self) -> None:
        pipeline = Krea2Pipeline()
        txt = torch.ones(1, 1, 1, dtype=torch.bfloat16)

        for index in range(pipeline.CONDITIONING_CACHE_MAX + 1):
            pipeline._put_conditioning_cache(("key", index), txt, None)

        self.assertIsNone(pipeline._get_conditioning_cache(("key", 0)))
        self.assertIsNotNone(
            pipeline._get_conditioning_cache(("key", pipeline.CONDITIONING_CACHE_MAX))
        )


if __name__ == "__main__":
    unittest.main()
