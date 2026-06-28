from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class _TinyDiT(nn.Module):
    def __init__(self, n_blocks: int = 6, dim: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class BlockSwapPlanTests(unittest.TestCase):
    def test_swap_plan_targets_last_n_blocks(self) -> None:
        from krea2.block_swap import resolve_swap_plan

        plan = resolve_swap_plan(total_blocks=28, blocks_to_swap=8)

        self.assertEqual(plan.swapped_indices, list(range(20, 28)))
        self.assertEqual(plan.resident_count, 20)

    def test_swap_plan_clamps_to_total(self) -> None:
        from krea2.block_swap import resolve_swap_plan

        plan = resolve_swap_plan(total_blocks=6, blocks_to_swap=999)

        self.assertEqual(plan.swapped_indices, list(range(0, 6)))
        self.assertEqual(plan.resident_count, 0)

    def test_zero_swap_is_noop_plan(self) -> None:
        from krea2.block_swap import resolve_swap_plan

        plan = resolve_swap_plan(total_blocks=6, blocks_to_swap=0)

        self.assertEqual(plan.swapped_indices, [])
        self.assertEqual(plan.resident_count, 6)


class BlockSwapControllerTests(unittest.TestCase):
    def test_forward_output_is_unchanged_with_swap_installed(self) -> None:
        from krea2.block_swap import BlockSwapController

        torch.manual_seed(0)
        model = _TinyDiT(n_blocks=6).eval()
        x = torch.randn(2, 4)
        with torch.no_grad():
            baseline = model(x).clone()

        controller = BlockSwapController(
            model, blocks_to_swap=3, device="cpu", offload_device="cpu", prefetch=0, pin_memory=False
        )
        controller.install()
        try:
            with torch.no_grad():
                swapped_out = model(x).clone()
        finally:
            controller.remove()

        self.assertTrue(torch.allclose(baseline, swapped_out, atol=1e-6))

    def test_each_swapped_block_moves_in_and_out_once_per_forward(self) -> None:
        from krea2.block_swap import BlockSwapController

        model = _TinyDiT(n_blocks=6).eval()
        controller = BlockSwapController(
            model, blocks_to_swap=2, device="cpu", offload_device="cpu", prefetch=0, pin_memory=False
        )
        controller.install()
        try:
            with torch.no_grad():
                model(torch.randn(1, 4))
        finally:
            controller.remove()

        self.assertEqual(controller.stats["swaps_in"], 2)
        self.assertEqual(controller.stats["swaps_out"], 2)

    def test_remove_restores_all_blocks_and_clears_hooks(self) -> None:
        from krea2.block_swap import BlockSwapController

        model = _TinyDiT(n_blocks=5).eval()
        controller = BlockSwapController(
            model, blocks_to_swap=2, device="cpu", offload_device="cpu", prefetch=0, pin_memory=False
        )
        controller.install()
        controller.remove()

        for block in model.blocks:
            self.assertEqual(len(block._forward_pre_hooks), 0)
            self.assertEqual(len(block._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
