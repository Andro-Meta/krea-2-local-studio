from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from quality_assets import asset_by_id, asset_installed, asset_specs, download_asset  # noqa: E402
from settings import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Redraw quality benchmark/provider assets.")
    parser.add_argument(
        "--assets",
        default="krea2_turbo_bf16,krea2_raw_bf16,flux_fill",
        help="Comma-separated asset ids, or 'all'.",
    )
    parser.add_argument("--check", action="store_true", help="Only report local status.")
    return parser.parse_args()


def selected_specs(value: str):
    specs = asset_specs()
    if value == "all":
        return specs
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    return [asset_by_id(item) for item in wanted]


def main() -> int:
    # This script is explicitly for downloading, so override session-level offline
    # mode without permanently changing the user's environment.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    args = parse_args()
    specs = selected_specs(args.assets)
    token = settings.hf_token or os.environ.get("HF_TOKEN") or None

    failed: list[str] = []
    for spec in specs:
        installed = asset_installed(spec)
        mark = "OK" if installed else "MISSING"
        print(f"[{mark}] {spec.id}: {spec.purpose}")
        print(f"      repo={spec.repo_id}")
        print(f"      local={spec.local_path}")
        if args.check or installed:
            continue
        try:
            path = download_asset(spec, token=token)
            print(f"      downloaded -> {path}")
        except Exception as exc:
            failed.append(spec.id)
            print(f"      ERROR: {exc}")

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
