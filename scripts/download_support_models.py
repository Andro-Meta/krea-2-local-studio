from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from support_models import download_support_models, support_model_status  # noqa: E402


def print_status() -> bool:
    statuses = support_model_status()
    ok = True
    print("Krea support models:")
    for item in statuses:
        installed = bool(item["installed"])
        ok = ok and installed
        mark = "OK" if installed else "MISSING"
        print(f"  [{mark}] {item['label']} ({item['repo_id']})")
        print(f"       {item['purpose']}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check/download Krea support models.")
    parser.add_argument("--check", action="store_true", help="Only check local cache status.")
    args = parser.parse_args()

    if args.check:
        return 0 if print_status() else 1

    if print_status():
        print("\nSupport models already present.")
        return 0

    print("\nDownloading support models. Qwen3-VL is large and may take a while...")
    try:
        results = download_support_models()
    except Exception as exc:
        print(f"ERROR: Support model download failed: {exc}")
        return 1

    print("\nDownload results:")
    for result in results:
        mark = "OK" if result["installed"] else "MISSING"
        print(f"  [{mark}] {result['repo_id']} -> {result['path']}")

    return 0 if print_status() else 1


if __name__ == "__main__":
    raise SystemExit(main())
