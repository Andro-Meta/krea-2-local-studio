from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GGUF realtime mode after GGUF txt2img runtime is configured.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print({
        "dry_run": bool(args.dry_run),
        "enabled": False,
        "message": "GGUF realtime is intentionally disabled until GGUF txt2img benchmark and timeout validation pass.",
    })
    return 0 if args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
