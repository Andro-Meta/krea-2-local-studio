from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised by lightweight CI
    raise unittest.SkipTest("torch is not installed in the lightweight CI environment") from exc

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


class AttentionBackendTests(unittest.TestCase):
    def test_default_attention_backend_is_sdpa(self) -> None:
        from krea2 import mmdit

        self.assertEqual(mmdit.attention_backend(), "sdpa")

    def test_sage_request_falls_back_to_sdpa_on_cpu(self) -> None:
        from krea2 import mmdit

        old_backend = mmdit.KREA_ATTENTION_BACKEND
        try:
            mmdit.KREA_ATTENTION_BACKEND = "sage"
            q = torch.randn(1, 2, 4, 8)
            k = torch.randn(1, 2, 4, 8)
            v = torch.randn(1, 2, 4, 8)

            out = mmdit.attention(q, k, v)

            self.assertEqual(out.shape, (1, 4, 16))
        finally:
            mmdit.KREA_ATTENTION_BACKEND = old_backend


if __name__ == "__main__":
    unittest.main()
