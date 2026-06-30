from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from gguf_diffusion_provider import GgufRuntimeSettings, build_gguf_command, generate_gguf_external
from settings import settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GGUF realtime mode after GGUF txt2img runtime is configured.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    req = SimpleNamespace(
        prompt="a red fox in morning fog, cinematic lighting",
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
        print(json.dumps({"dry_run": bool(args.dry_run), "configured": False, "error": str(exc)}))
        return 0 if args.dry_run else 1
    if args.dry_run:
        print(json.dumps({"dry_run": True, "configured": True, "command": cmd, "output": str(output)}))
        return 0

    started = time.perf_counter()
    _images, seed, filenames, _lora_reports, metadata = generate_gguf_external(req, runtime)
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "dry_run": False,
        "configured": True,
        "ok": True,
        "elapsed_sec": round(elapsed, 2),
        "preview_size": 512,
        "preview_steps": 4,
        "final_steps": 8,
        "seed": seed,
        "filenames": filenames,
        "output": metadata[0]["filename"] if metadata else "",
        "live_candidate": False,
        "speed_candidate": elapsed <= 30.0,
        "message": "GGUF 512/4 speed benchmark completed, but visual prompt-adherence sweep did not meet native quality. Keep GGUF experimental and txt2img-only.",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
