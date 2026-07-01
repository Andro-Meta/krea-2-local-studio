from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class NativeGgufLoaderTests(unittest.TestCase):
    def test_real_krea_gguf_has_full_native_key_coverage(self) -> None:
        from krea2.gguf_quant import inspect_gguf_for_krea

        path = ROOT / "models" / "gguf" / "Krea-2-Turbo-Q4_K_M.gguf"
        if not path.exists():
            self.skipTest("local Krea GGUF fixture is not installed")

        info = inspect_gguf_for_krea(path)

        self.assertEqual(info["missing_native_keys"], [])
        self.assertIn("last.down.weight", info["extra_gguf_keys"])
        self.assertGreaterEqual(info["tensor_count"], 430)

    def test_loader_dequantizes_and_fp8_quantizes_big_linears(self) -> None:
        from gguf import GGMLQuantizationType
        from krea2.gguf_quant import load_gguf_as_fp8_scaled

        class FakeTensor:
            def __init__(self, name, array, qtype=GGMLQuantizationType.F32):
                self.name = name
                self.array = np.asarray(array, dtype=np.float32)
                self.data = self
                self.tensor_type = qtype

        class FakeReader:
            def __init__(self, _path):
                self.tensors = [
                    FakeTensor("big.weight", np.arange(256, dtype=np.float32).reshape(16, 16)),
                    FakeTensor("bias", np.arange(4, dtype=np.float32)),
                    FakeTensor("extra.weight", np.ones((2, 2), dtype=np.float32)),
                ]

        def fake_dequant(data, _qtype):
            return data.array if hasattr(data, "array") else data

        with patch("krea2.gguf_quant.GGUFReader", FakeReader), patch("krea2.gguf_quant.dequantize", fake_dequant):
            sd, scales = load_gguf_as_fp8_scaled(
                "fake.gguf",
                target_shapes={"big.weight": (16, 16), "bias": (4,)},
                min_numel=128,
                device="cpu",
                compute_dtype=torch.bfloat16,
            )

        self.assertEqual(sd["big.weight"].dtype, torch.float8_e4m3fn)
        self.assertIn("big", scales)
        self.assertEqual(sd["bias"].dtype, torch.bfloat16)
        self.assertNotIn("extra.weight", sd)

    def test_preflight_treats_gguf_as_low_vram_native_storage(self) -> None:
        import tempfile

        import inference

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Krea-2-Turbo-Q4_K_M.gguf"
            path.write_bytes(b"gguf")

            with patch.object(inference, "get_ram_gb", return_value=(16.0, 12.0)), \
                 patch.object(inference, "get_gpu_info", return_value=("RTX 4090", 24.0, 22.0)), \
                 patch.object(inference, "get_gpu_process_details", return_value=[]):
                inference.preflight_model_load(path, "gguf")


if __name__ == "__main__":
    unittest.main()
