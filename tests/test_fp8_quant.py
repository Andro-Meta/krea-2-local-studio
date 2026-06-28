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


class Fp8ScaledQuantTests(unittest.TestCase):
    def test_quantize_linear_weight_round_trips_within_tolerance(self) -> None:
        from krea2.fp8_quant import quantize_weight_to_fp8_scaled

        torch.manual_seed(0)
        weight = torch.randn(64, 128, dtype=torch.bfloat16) * 0.05
        qweight, scale = quantize_weight_to_fp8_scaled(weight)

        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        dequant = qweight.to(torch.float32) * scale
        rel_err = (dequant - weight.float()).abs().mean() / weight.float().abs().mean().clamp_min(1e-8)
        self.assertLess(rel_err.item(), 0.1)

    def test_scale_is_absmax_over_fp8_max(self) -> None:
        from krea2.fp8_quant import FP8_E4M3_MAX, quantize_weight_to_fp8_scaled

        weight = torch.tensor([[0.0, 0.5, -1.0, 2.0]], dtype=torch.float32)
        _qweight, scale = quantize_weight_to_fp8_scaled(weight)

        expected = 2.0 / FP8_E4M3_MAX
        self.assertAlmostEqual(scale, expected, places=6)

    def test_zero_weight_uses_safe_unit_scale(self) -> None:
        from krea2.fp8_quant import quantize_weight_to_fp8_scaled

        weight = torch.zeros(8, 8, dtype=torch.float32)
        qweight, scale = quantize_weight_to_fp8_scaled(weight)

        self.assertGreater(scale, 0.0)
        self.assertTrue(torch.all(qweight.to(torch.float32) == 0))

    def test_quantize_state_dict_only_targets_2d_linear_weights(self) -> None:
        from krea2.fp8_quant import quantize_linear_weights_to_fp8_scaled

        sd = {
            "blocks.0.attn.weight": torch.randn(32, 32, dtype=torch.bfloat16),
            "blocks.0.attn.bias": torch.randn(32, dtype=torch.bfloat16),
            "norm.weight": torch.randn(32, dtype=torch.bfloat16),
        }
        out_sd, scales = quantize_linear_weights_to_fp8_scaled(sd, min_numel=256)

        self.assertEqual(out_sd["blocks.0.attn.weight"].dtype, torch.float8_e4m3fn)
        self.assertIn("blocks.0.attn", scales)
        self.assertEqual(out_sd["blocks.0.attn.bias"].dtype, torch.bfloat16)
        self.assertEqual(out_sd["norm.weight"].dtype, torch.bfloat16)
        self.assertNotIn("norm", scales)

    def test_streaming_loader_quantizes_big_weights_from_safetensors(self) -> None:
        import tempfile

        from safetensors.torch import save_file

        from krea2.fp8_quant import load_bf16_as_fp8_scaled

        sd = {
            "blocks.0.proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
            "blocks.0.proj.bias": torch.randn(64, dtype=torch.bfloat16),
            "small.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.safetensors"
            save_file(sd, str(path))

            out_sd, scales = load_bf16_as_fp8_scaled(path, min_numel=256)

        self.assertEqual(out_sd["blocks.0.proj.weight"].dtype, torch.float8_e4m3fn)
        self.assertIn("blocks.0.proj", scales)
        self.assertEqual(out_sd["small.weight"].dtype, torch.bfloat16)
        self.assertNotIn("small", scales)

    def test_streaming_loader_casts_float32_layers_to_compute_dtype(self) -> None:
        import tempfile

        from safetensors.torch import save_file

        from krea2.fp8_quant import load_bf16_as_fp8_scaled

        # A checkpoint mixing a big quantizable weight with a small float32 layer
        # (which would otherwise mismatch bf16 activations in a Linear).
        sd = {
            "blocks.0.proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
            "blocks.0.mod.weight": torch.randn(8, 8, dtype=torch.float32),
            "blocks.0.mod.bias": torch.randn(8, dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.safetensors"
            save_file(sd, str(path))

            out_sd, _ = load_bf16_as_fp8_scaled(path, min_numel=256, compute_dtype=torch.bfloat16)

        self.assertEqual(out_sd["blocks.0.proj.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(out_sd["blocks.0.mod.weight"].dtype, torch.bfloat16)
        self.assertEqual(out_sd["blocks.0.mod.bias"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
