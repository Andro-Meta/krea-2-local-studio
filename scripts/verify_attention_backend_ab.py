from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2 import mmdit


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Krea attention backend selection and basic output shape.")
    parser.add_argument("--backend", choices=["sdpa", "sage"], default="sdpa")
    args = parser.parse_args()

    old_backend = mmdit.KREA_ATTENTION_BACKEND
    try:
        mmdit.KREA_ATTENTION_BACKEND = args.backend
        torch.manual_seed(123)
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        out = mmdit.attention(q, k, v)
        result = {
            "requested_backend": args.backend,
            "effective_backend": mmdit.attention_backend(),
            "device": str(q.device),
            "shape": list(out.shape),
            "finite": bool(torch.isfinite(out).all().item()),
        }
        print(json.dumps(result, indent=2))
        return 0 if result["finite"] and result["shape"] == [1, 8, 32] else 1
    finally:
        mmdit.KREA_ATTENTION_BACKEND = old_backend


if __name__ == "__main__":
    raise SystemExit(main())
