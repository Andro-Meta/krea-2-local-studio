from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "verify_krea2_node_parity"


def _wait_job(base_url: str, job_id: str, timeout: int = 900) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = requests.get(f"{base_url}/api/generate/{job_id}", timeout=30).json()
        if data.get("status") in {"done", "error"}:
            return data
        time.sleep(2)
    raise TimeoutError(f"job {job_id} timed out")


def _write_image(name: str, image_b64: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    path.write_bytes(base64.b64decode(image_b64))
    return str(path)


def _generate(base_url: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
    response.raise_for_status()
    job = _wait_job(base_url, response.json()["job_id"])
    if job.get("status") != "done":
        raise RuntimeError(f"{name} failed: {job.get('error')}")
    image = (job.get("images") or [None])[0]
    if image:
        job["saved_path"] = _write_image(name, image)
    return job


def main() -> int:
    parser = argparse.ArgumentParser(description="Live visual harness for Krea 2 node parity controls.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-generation", action="store_true", help="Only check system diagnostics.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    system = requests.get(f"{args.base_url}/api/system", timeout=30).json()
    summary: dict[str, Any] = {"system_attention": system.get("attention_acceleration"), "cases": []}

    cases = [
        ("baseline", {"prompt": "a cinematic glass fox in a mossy forest", "seed": 4242}),
        ("rebalance_subtle", {"prompt": "a cinematic glass fox in a mossy forest", "seed": 4242, "rebalance_preset": "subtle"}),
        ("rebalance_detail", {"prompt": "a cinematic glass fox in a mossy forest", "seed": 4242, "rebalance_preset": "detail"}),
        ("enhancer_capped", {
            "prompt": "a detailed brass robot portrait, sharp eyes, studio lighting",
            "seed": 5252,
            "krea_enhancer_enabled": True,
            "krea_enhancer_variant": "capped_delta",
            "krea_enhancer_delta_cap": 0.75,
        }),
        ("style_only", {
            "prompt": "a ceramic cabin, editorial product photo",
            "seed": 6262,
            "style_fusion_mode": "style_only",
        }),
    ]

    if not args.skip_generation:
        for name, patch in cases:
            payload = {
                "mode": "txt2img",
                "model_profile": "krea_turbo",
                "checkpoint": "turbo",
                "quantization": "fp8",
                "width": 768,
                "height": 768,
                "steps": 8,
                "cfg": 0,
                "num_images": 1,
                "sampler": "euler_flow",
                "scheduler": "simple",
                **patch,
            }
            job = _generate(args.base_url, name, payload)
            summary["cases"].append({"name": name, "status": job.get("status"), "saved_path": job.get("saved_path")})

    summary_path = OUT_DIR / "report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
