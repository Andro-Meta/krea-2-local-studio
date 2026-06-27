from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class KreaEnhancerTests(unittest.TestCase):
    def test_generation_request_defaults_leave_enhancer_off(self) -> None:
        from schemas import GenerationRequest

        data = GenerationRequest(prompt="a glass sculpture").model_dump()

        self.assertIn("krea_enhancer_enabled", data)
        self.assertFalse(data["krea_enhancer_enabled"])
        self.assertEqual(data["krea_enhancer_strength"], 1.0)
        self.assertEqual(data["krea_enhancer_variant"], "off")
        self.assertEqual(data["krea_enhancer_delta_cap"], 0.75)
        self.assertEqual(data["moodboard_ids"], [])

    def test_delta_cap_bounds_output_shift(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the lightweight CI environment")
        enhancer = importlib.import_module("krea_enhancer")

        class FakeTxtFusion:
            def original_forward(self, x, mask=None):
                return x[:, :, 0, :]

        txtfusion = FakeTxtFusion()
        txtfusion._krea_enhancer_original_forward = txtfusion.original_forward
        x = torch.ones((1, 1, 12, 2560), dtype=torch.float32)

        out = enhancer._enhanced_forward(txtfusion, x, mask=None, strength=1.0, variant="capped_delta", delta_cap=0.1)

        reference = txtfusion.original_forward(x)
        shift = torch.sqrt(torch.mean((out.float() - reference.float()) ** 2))
        base = torch.sqrt(torch.mean(reference.float() ** 2))
        self.assertLessEqual(float(shift / base), 0.1001)

    def test_shape_guard_skips_incompatible_txtfusion_input(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the lightweight CI environment")
        enhancer = importlib.import_module("krea_enhancer")

        class FakeTxtFusion:
            def original_forward(self, x, mask=None):
                return x.mean(dim=-1)

        txtfusion = FakeTxtFusion()
        txtfusion._krea_enhancer_original_forward = txtfusion.original_forward
        x = torch.ones((1, 1, 8, 128), dtype=torch.float32)

        out = enhancer._enhanced_forward(txtfusion, x, mask=None, strength=1.0, variant="capped_delta", delta_cap=0.1)

        self.assertTrue(torch.equal(out, txtfusion.original_forward(x)))

    def test_enhancer_context_temporarily_patches_txtfusion(self) -> None:
        if torch is None:
            self.skipTest("torch is not installed in the lightweight CI environment")
        enhancer = importlib.import_module("krea_enhancer")

        class FakeTxtFusion:
            def __init__(self):
                self.calls = 0
                self.forward = self.original_forward

            def original_forward(self, x, mask=None):
                self.calls += 1
                return x[:, :, 0, :]

        class FakeModel:
            def __init__(self):
                self.txtfusion = FakeTxtFusion()
                self.txtmlp = object()
                self.blocks = []
                self.txtlayers = 12
                self.txtdim = 2560

            def _unpack_context(self):
                return None

        model = FakeModel()
        original_forward = model.txtfusion.forward
        x = torch.ones((1, 2, 12, 2560), dtype=torch.float32)

        with enhancer.krea_enhancer_context(model, enabled=True, strength=1.0):
            self.assertIsNot(model.txtfusion.forward, original_forward)
            out = model.txtfusion.forward(x)
            self.assertEqual(tuple(out.shape), (1, 2, 2560))

        self.assertIs(model.txtfusion.forward, original_forward)
        self.assertGreater(model.txtfusion.calls, 1)


if __name__ == "__main__":
    unittest.main()
