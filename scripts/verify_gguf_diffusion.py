from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from gguf_diffusion_provider import GgufRuntimeSettings, build_gguf_command
from settings import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GGUF diffusion sidecar command construction.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    req = SimpleNamespace(
        prompt="a red fox in fog",
        negative_prompt="blurry",
        checkpoint="turbo",
        width=512,
        height=512,
        steps=4,
        cfg=0.0,
        seed=123,
        mode="txt2img",
    )
    runtime = GgufRuntimeSettings(
        sd_cli_path=settings.gguf_sd_cli_path,
        turbo_path=settings.gguf_turbo_path,
        raw_path=settings.gguf_raw_path,
        llm_path=settings.gguf_llm_path,
        vae_path=settings.gguf_vae_path,
        lora_dir=settings.gguf_lora_dir,
        timeout_sec=settings.gguf_timeout_sec,
    )
    try:
        cmd, output = build_gguf_command(req, runtime)
    except Exception as exc:
        print({"dry_run": bool(args.dry_run), "configured": False, "error": str(exc)})
        return 0 if args.dry_run else 1
    print({"dry_run": bool(args.dry_run), "configured": True, "command": cmd, "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
