from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from resource_manager import plan_parallel_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark or sanity-check Krea batch modes.")
    parser.add_argument("--quick", action="store_true", help="Only run estimator sanity checks; no server/model required.")
    parser.add_argument("--free-vram-gb", type=float, default=24.0)
    args = parser.parse_args()

    scenarios = [
        {"batch": 1, "width": 1024, "height": 1024, "cfg": 0.0, "mode": "txt2img", "checkpoint": "turbo"},
        {"batch": 2, "width": 1024, "height": 1024, "cfg": 0.0, "mode": "txt2img", "checkpoint": "turbo"},
        {"batch": 4, "width": 1024, "height": 1024, "cfg": 0.0, "mode": "txt2img", "checkpoint": "turbo"},
        {"batch": 2, "width": 2048, "height": 2048, "cfg": 0.0, "mode": "txt2img", "checkpoint": "turbo"},
    ]
    started = time.perf_counter()
    rows = []
    for scenario in scenarios:
        plan = plan_parallel_batch(
            free_vram_gb=args.free_vram_gb,
            width=scenario["width"],
            height=scenario["height"],
            quantization="fp8",
            batch=scenario["batch"],
            cfg_active=scenario["cfg"] > 0,
            mode=scenario["mode"],
            checkpoint=scenario["checkpoint"],
        )
        rows.append({**scenario, "allowed": plan["allowed"], "warnings": plan["warnings"]})
    print(json.dumps({"quick": bool(args.quick), "elapsed_sec": round(time.perf_counter() - started, 3), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
