from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class Int8ConvRotTests(unittest.TestCase):
    def test_convrot_int8_linear_matches_float_linear(self) -> None:
        from krea2.int8_convrot import NativeInt8Linear, build_hadamard, quantize_int8_axiswise, rotate_weight

        torch.manual_seed(123)
        weight = torch.randn(5, 16, dtype=torch.float32) * 0.25
        bias = torch.randn(5, dtype=torch.float32) * 0.05
        x = torch.randn(19, 16, dtype=torch.float32)
        h = build_hadamard(16, dtype=torch.float32)
        rotated = rotate_weight(weight, h, group_size=16)
        qweight, scale = quantize_int8_axiswise(rotated, dim=1)

        layer = NativeInt8Linear(16, 5, bias=True, compute_dtype=torch.float32)
        layer.load_quantized_weight(
            qweight,
            scale,
            bias=bias,
            convrot=True,
            convrot_groupsize=16,
        )

        actual = layer(x)
        expected = torch.nn.functional.linear(x, weight, bias)

        rel_err = (actual - expected).abs().mean() / expected.abs().mean().clamp_min(1e-8)
        self.assertLess(float(rel_err), 0.02)

    def test_inspects_comfy_style_safetensors_metadata(self) -> None:
        from safetensors.torch import save_file

        from krea2.int8_convrot import inspect_int8_safetensors

        quant_conf = json.dumps({"format": "int8", "per_row": True, "convrot": True, "convrot_groupsize": 256}).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny_int8_convrot.safetensors"
            save_file(
                {
                    "blocks.0.attn.wq.weight": torch.zeros((4, 8), dtype=torch.int8),
                    "blocks.0.attn.wq.weight_scale": torch.ones((4, 1), dtype=torch.float32),
                    "blocks.0.attn.wq.comfy_quant": torch.tensor(list(quant_conf), dtype=torch.uint8),
                    "blocks.0.attn.wq.input_scale": torch.ones((1,), dtype=torch.float32),
                    "first.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
                },
                str(path),
                metadata={"source": "unit-test"},
            )

            info = inspect_int8_safetensors(path)

        self.assertTrue(info["ok"])
        self.assertEqual(info["tensor_count"], 5)
        self.assertEqual(info["int8_layer_count"], 1)
        self.assertEqual(info["convrot_layer_count"], 1)
        self.assertEqual(info["int8_layers"][0]["name"], "blocks.0.attn.wq")
        self.assertEqual(info["int8_layers"][0]["convrot_groupsize"], 256)
        self.assertIn("first", info["high_precision_prefixes_present"])

    def test_replaces_only_quantized_linear_modules(self) -> None:
        from krea2.int8_convrot import NativeInt8Linear, replace_int8_linears

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.keep = torch.nn.Linear(8, 8)
                self.blocks = torch.nn.ModuleList([torch.nn.Module()])
                self.blocks[0].proj = torch.nn.Linear(8, 4)

        model = Tiny()
        replaced = replace_int8_linears(model, {"blocks.0.proj": {"convrot": True, "convrot_groupsize": 256}})

        self.assertEqual(replaced, 1)
        self.assertIsInstance(model.blocks[0].proj, NativeInt8Linear)
        self.assertIsInstance(model.keep, torch.nn.Linear)
        self.assertNotIsInstance(model.keep, NativeInt8Linear)

    def test_existing_lora_wrapper_composes_with_native_int8_linear(self) -> None:
        from krea2.int8_convrot import NativeInt8Linear, quantize_int8_axiswise
        from lora_manager import _ensure_lora_wrapper

        weight = torch.randn(3, 8, dtype=torch.float32) * 0.2
        qweight, scale = quantize_int8_axiswise(weight, dim=1)
        layer = NativeInt8Linear(8, 3, bias=False, compute_dtype=torch.float32)
        layer.load_quantized_weight(qweight, scale)
        x = torch.randn(17, 8, dtype=torch.float32)

        base = layer(x)
        a = torch.randn(2, 8, dtype=torch.float32) * 0.1
        b = torch.randn(3, 2, dtype=torch.float32) * 0.1
        _ensure_lora_wrapper(layer)
        layer._lora_adapters.append((a, b, 0.75))

        actual = layer(x)
        expected = base + 0.75 * torch.nn.functional.linear(torch.nn.functional.linear(x, a), b)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
