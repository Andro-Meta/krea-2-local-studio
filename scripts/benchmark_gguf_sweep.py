from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from gguf_diffusion_provider import GgufRuntimeSettings, generate_gguf_external
from settings import settings


def main() -> int:
    prompt = "a sad clown in a vacant cyberpunk cityscape"
    variants = [
        ("q4_512_4", settings.gguf_turbo_path, 512, 4),
        ("q4_512_8", settings.gguf_turbo_path, 512, 8),
        ("q3_512_4", str(ROOT / "models" / "gguf" / "Krea-2-Turbo-Q3_K_M.gguf"), 512, 4),
        ("q3_512_8", str(ROOT / "models" / "gguf" / "Krea-2-Turbo-Q3_K_M.gguf"), 512, 8),
        ("q4_1024_8", settings.gguf_turbo_path, 1024, 8),
    ]
    results = []
    for name, model_path, size, steps in variants:
        req = SimpleNamespace(
            prompt=prompt,
            negative_prompt="",
            checkpoint="turbo",
            width=size,
            height=size,
            steps=steps,
            cfg=0.0,
            seed=4001,
            mode="txt2img",
        )
        runtime = GgufRuntimeSettings(
            sd_cli_path=settings.gguf_sd_cli_path,
            turbo_path=model_path,
            raw_path=settings.gguf_raw_path,
            llm_path=settings.gguf_llm_path,
            vae_path=settings.gguf_vae_path,
            lora_dir=settings.gguf_lora_dir,
            timeout_sec=settings.gguf_timeout_sec,
        )
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        try:
            _images, seed, filenames, _reports, metadata = generate_gguf_external(req, runtime)
            elapsed = round(time.perf_counter() - started, 2)
            output = metadata[0]["filename"] if metadata else (filenames[0] if filenames else "")
            row = {
                "name": name,
                "status": "done",
                "elapsed_sec": elapsed,
                "output": output,
                "size": size,
                "steps": steps,
                "model": Path(model_path).name,
                "seed": seed,
            }
        except Exception as exc:
            row = {
                "name": name,
                "status": "error",
                "error": str(exc),
                "size": size,
                "steps": steps,
                "model": Path(model_path).name,
            }
        print(json.dumps(row), flush=True)
        results.append(row)

    out = ROOT / "outputs" / "gguf_sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SWEEP_DONE")
    print(json.dumps(results, indent=2))
    return 0 if all(row["status"] == "done" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
