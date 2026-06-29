from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from prompt_expander import expand_prompt_result
from settings import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the configured GGUF helper OpenAI-compatible endpoint.")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = expand_prompt_result(
        "a red fox in fog",
        backend="gguf-server",
        gguf_helper_base_url=settings.gguf_helper_base_url,
        gguf_helper_model=settings.gguf_helper_model,
        gguf_helper_timeout_sec=settings.gguf_helper_timeout_sec,
    )
    print({"backend": result.backend, "changed": result.changed, "error": result.error, "preview": result.expanded[:160]})
    return 0 if args.quick else (1 if result.error else 0)


if __name__ == "__main__":
    raise SystemExit(main())
