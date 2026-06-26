from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import os
import time
from pathlib import Path

from verify_redraw_studio_workflows import (
    ROOT,
    find_base,
    make_inpaint_mask,
    make_mood,
    make_object,
    make_person,
    make_scene,
    make_sketch,
    make_style,
    request_json,
    wait_job,
    wait_model,
    workflow_payloads,
)


OUT_DIR = ROOT / "outputs" / f"benchmark_redraw_quality_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

VARIANTS = {
    "turbo-fp8": {
        "checkpoint": "turbo",
        "quantization": "fp8",
        "path_env": "KREA2_TURBO_FP8_PATH",
        "default_path": ROOT / "models" / "krea2" / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors",
        "steps": 8,
        "cfg": 0,
        "mu": 1.15,
        "max_size": 912,
    },
    "turbo-bf16": {
        "checkpoint": "turbo",
        "quantization": "bf16",
        "path_env": "KREA2_TURBO_BF16_PATH",
        "default_path": ROOT / "models" / "krea2" / "diffusion_models" / "krea2_turbo_bf16.safetensors",
        "steps": 8,
        "cfg": 0,
        "mu": 1.15,
        "max_size": 912,
    },
    "raw-bf16": {
        "checkpoint": "raw",
        "quantization": "bf16",
        "path_env": "KREA2_RAW_BF16_PATH",
        "default_path": ROOT / "models" / "krea2" / "diffusion_models" / "krea2_raw_bf16.safetensors",
        "steps": 52,
        "cfg": 3.5,
        "mu": None,
        "max_size": 1024,
    },
}


def model_path(config: dict) -> Path:
    return Path(os.environ.get(config["path_env"], str(config["default_path"])))


def ensure_variant_loaded(base: str, variant: str, config: dict) -> bool:
    path = model_path(config)
    if not path.exists():
        return False
    status = request_json(base, "/api/system").get("model_status", {})
    if (
        status.get("loaded")
        and status.get("checkpoint") == config["checkpoint"]
        and status.get("quantization") == config["quantization"]
    ):
        return True
    request_json(
        base,
        "/api/load-model",
        method="POST",
        payload={"checkpoint_path": str(path), "quantization": config["quantization"]},
    )
    wait_model(base)
    print(f"MODEL_READY {variant}: {path}", flush=True)
    return True


def references() -> dict[str, str]:
    return {
        "scene": make_scene(),
        "person": make_person(),
        "object": make_object(),
        "sketch": make_sketch(),
        "style": make_style(),
        "mood": make_mood(),
        "mask": make_inpaint_mask(),
    }


def save_ref(name: str, image_b64: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"ref_{name}.png"
    path.write_bytes(base64.b64decode(image_b64))
    return str(path)


def apply_variant(payload: dict, config: dict) -> dict:
    tuned = dict(payload)
    tuned["checkpoint"] = config["checkpoint"]
    tuned["quantization"] = config["quantization"]
    tuned["steps"] = config["steps"]
    tuned["cfg"] = config["cfg"]
    tuned["mu"] = config["mu"]
    tuned["seed"] = 260626
    tuned["quality_preset"] = "raw_benchmark" if config["checkpoint"] == "raw" else "balanced"
    if config["checkpoint"] == "raw":
        tuned["width"] = min(int(tuned.get("width", 512)), config["max_size"])
        tuned["height"] = min(int(tuned.get("height", 512)), config["max_size"])
    return tuned


def run_payload(base: str, label: str, payload: dict) -> dict:
    start = time.perf_counter()
    job = request_json(base, "/api/generate", method="POST", payload=payload)
    result = wait_job(base, job["job_id"])
    elapsed = round(time.perf_counter() - start, 2)
    if result.get("status") != "done" or not result.get("images"):
        return {
            "status": "error",
            "elapsed_s": elapsed,
            "error": result.get("error") or result,
            "payload": payload,
        }
    output_path = OUT_DIR / f"{label}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(result["images"][0]))
    return {
        "status": "needs_review",
        "elapsed_s": elapsed,
        "seed": result.get("seed"),
        "file": str(output_path),
        "payload": payload,
        "notes": "",
    }


def write_report(summary: dict) -> str:
    cards = []
    for variant, tasks in summary["variants"].items():
        for task, item in tasks.items():
            rel = Path(item["file"]).name if item.get("file") else ""
            img = f'<img src="{html.escape(rel)}" loading="lazy">' if rel else ""
            cards.append(
                "<section>"
                f"<h3>{html.escape(variant)} / {html.escape(task)}</h3>"
                f"{img}"
                f"<p>Status: {html.escape(str(item.get('status')))} | {html.escape(str(item.get('elapsed_s')))}s</p>"
                f"<p>Visual review: {html.escape(str(item.get('visual_review', 'needs_review')))}</p>"
                "</section>"
            )
    body = "\n".join(cards)
    path = OUT_DIR / "contact_sheet.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Redraw Quality Benchmark</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#111;color:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}"
        "section{background:#1d1d1d;border:1px solid #333;border-radius:16px;padding:12px}"
        "img{width:100%;border-radius:12px;background:#333}p{color:#bbb;font-size:13px}</style>"
        f"<h1>Redraw Quality Benchmark</h1><main>{body}</main>",
        encoding="utf-8",
    )
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Redraw Studio quality across Krea model variants.")
    parser.add_argument("--variants", default="turbo-fp8,turbo-bf16,raw-bf16")
    parser.add_argument("--tasks", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    requested_tasks = None if args.tasks == "all" else {t.strip() for t in args.tasks.split(",") if t.strip()}
    base = find_base()
    refs = references()
    for name, image_b64 in refs.items():
        save_ref(name, image_b64)
    workflows = workflow_payloads(refs)

    summary = {"base_url": base, "created_at": dt.datetime.now().isoformat(), "variants": {}, "skipped": {}}
    for variant in requested_variants:
        config = VARIANTS.get(variant)
        if not config:
            summary["skipped"][variant] = "Unknown variant."
            continue
        if not ensure_variant_loaded(base, variant, config):
            summary["skipped"][variant] = f"Checkpoint not found: {model_path(config)}"
            continue
        summary["variants"][variant] = {}
        for task, payload in workflows.items():
            if requested_tasks is not None and task not in requested_tasks:
                continue
            label = f"{variant}_{task}"
            print(f"BENCHMARK {label}", flush=True)
            summary["variants"][variant][task] = run_payload(base, label, apply_variant(payload, config))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary["contact_sheet"] = write_report(summary)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "contact_sheet": summary["contact_sheet"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
