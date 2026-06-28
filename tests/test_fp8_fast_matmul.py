import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    import torch
except Exception as exc:  # pragma: no cover - torch absent in lightweight CI
    raise unittest.SkipTest(f"torch unavailable: {exc}")

if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA required for fp8 _scaled_mm")

_FP8 = getattr(torch, "float8_e4m3fn", None)
if _FP8 is None:
    raise unittest.SkipTest("float8_e4m3fn unsupported")


def _supports_fp8_compute() -> bool:
    props = torch.cuda.get_device_properties(0)
    return props.major >= 9 or (props.major == 8 and props.minor >= 9)


class TestFp8ScaledMmConvention(unittest.TestCase):
    """Validate the scale convention the fast-matmul forward relies on:

    _scaled_mm(x_fp8, w_fp8.t(), scale_a=1, scale_b=s) == x @ (s * w_fp8).t()
    i.e. it reproduces a regular linear against the dequantized bf16 weight.
    """

    def setUp(self):
        if not _supports_fp8_compute():
            self.skipTest("GPU lacks fp8 tensor-core compute (Ada/Hopper/Blackwell)")
        torch.manual_seed(0)

    def test_scaled_mm_matches_dequant_linear(self):
        dev = "cuda"
        K, N, M = 256, 128, 64  # 16-aligned for fp8 matmul
        w_bf16 = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
        scale = w_bf16.abs().max().float() / 448.0
        w_fp8 = (w_bf16.float() / scale).clamp(-448, 448).to(_FP8)
        # Dequantized reference weight.
        w_deq = w_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)

        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
        ref = torch.nn.functional.linear(x, w_deq)

        xin = torch.clamp(x, -448, 448).to(_FP8).contiguous()
        out = torch._scaled_mm(
            xin, w_fp8.t(),
            out_dtype=torch.bfloat16,
            scale_a=torch.ones((), device=dev, dtype=torch.float32),
            scale_b=scale.to(device=dev, dtype=torch.float32),
        )
        if isinstance(out, tuple):
            out = out[0]

        # fp8 activation quantization is lossy; check correlation + bounded error.
        rel = (out.float() - ref.float()).abs().mean() / (ref.float().abs().mean() + 1e-6)
        self.assertLess(rel.item(), 0.15)


if __name__ == "__main__":
    unittest.main()
