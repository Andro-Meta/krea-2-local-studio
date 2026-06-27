from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from memory_manager import cleanup_krea_runtime_processes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop stale Krea runtime processes before startup.")
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = cleanup_krea_runtime_processes(wait_seconds=args.wait_seconds)
    if args.json:
        print(json.dumps(result))
    else:
        stopped = ", ".join(str(pid) for pid in result["stopped_pids"]) or "none"
        print(f"Stopped stale Krea processes: {stopped}")
        if result["remaining"]:
            remaining = ", ".join(str(proc["pid"]) for proc in result["remaining"])
            print(f"WARNING: Krea processes still running after cleanup: {remaining}")
        print(result["memory"])
    return 1 if result["remaining"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
