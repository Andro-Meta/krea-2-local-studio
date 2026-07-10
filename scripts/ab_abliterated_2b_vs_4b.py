"""A/B Huihui 4B vs 2B Instruct abliterated via ComfyUI-QwenVL.

Usage:
  venv\\Scripts\\python.exe scripts\\ab_abliterated_2b_vs_4b.py [image_path]
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from prompt_expander import EXPANSION_SYSTEM_PROMPT, _describe_prompt  # noqa: E402
from comfy_qwen_vl import (  # noqa: E402
    DEFAULT_MODEL,
    QUANT_FP16,
    QWEN_VL_NODE,
    _attach_preview,
    _b64_to_png_bytes,
    _ensure_nodes,
    _link,
    _run_graph_for_text,
    _upload_png,
    comfy_qwen_vl_available,
)


MODELS = (
    "Huihui-Qwen3-VL-4B-Instruct-abliterated",
    "Huihui-Qwen3-VL-2B-Instruct-abliterated",
)


def _find_image(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.is_file():
            return p
        raise SystemExit(f"Image not found: {p}")
    for folder in (ROOT / "outputs", ROOT / "data" / "custom_moodboards"):
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and p.is_file():
                return p
    raise SystemExit("No sample image found")


def _run_describe(model_name: str, image_b64: str, prompt: str, *, seed: int = 42) -> str:
    _ensure_nodes()
    png = _b64_to_png_bytes(image_b64, max_side=1024)
    fname = _upload_png(png)
    nodes = {
        "load": {"class_type": "LoadImage", "inputs": {"image": fname}},
        "qwen": {
            "class_type": QWEN_VL_NODE,
            "inputs": {
                "model_name": model_name,
                "quantization": QUANT_FP16,
                "attention_mode": "sdpa",
                "use_torch_compile": False,
                "device": "auto",
                "preset_prompt": "🖼️ Detailed Description",
                "custom_prompt": prompt,
                "max_tokens": 420,
                "temperature": 0.6,
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "frame_count": 1,
                "keep_model_loaded": False,
                "seed": seed,
                "image": _link("load"),
            },
        },
    }
    _attach_preview(nodes, "qwen")
    return _run_graph_for_text(nodes, "preview")


def _run_wand(model_name: str, user_prompt: str, *, seed: int = 42) -> str:
    _ensure_nodes()
    merged = f"{EXPANSION_SYSTEM_PROMPT.strip()}\n\n{user_prompt.strip()}".strip()
    nodes = {
        "qwen": {
            "class_type": QWEN_VL_NODE,
            "inputs": {
                "model_name": model_name,
                "quantization": QUANT_FP16,
                "attention_mode": "sdpa",
                "use_torch_compile": False,
                "device": "auto",
                "preset_prompt": "🖼️ Detailed Description",
                "custom_prompt": merged,
                "max_tokens": 700,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "frame_count": 1,
                "keep_model_loaded": False,
                "seed": seed,
            },
        }
    }
    _attach_preview(nodes, "qwen")
    return _run_graph_for_text(nodes, "preview")


def main() -> None:
    img_path = _find_image(sys.argv[1] if len(sys.argv) > 1 else None)
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    describe_prompt = _describe_prompt("recreate")
    wand_prompt = "a lonely lighthouse on a cliff at dusk"
    report: dict = {
        "image": str(img_path),
        "comfy_qwenvl_available": comfy_qwen_vl_available(),
        "default_helper_model": DEFAULT_MODEL,
        "describe_instruction": describe_prompt,
        "wand_prompt": wand_prompt,
        "models": {},
    }
    print(f"image={img_path}")
    print(f"comfy_qwenvl_available={report['comfy_qwenvl_available']}")

    for model in MODELS:
        print(f"\n===== {model} =====")
        entry: dict = {}
        t0 = time.time()
        try:
            text = _run_describe(model, b64, describe_prompt)
            entry["describe"] = {
                "seconds": round(time.time() - t0, 2),
                "chars": len(text or ""),
                "prompt": text,
                "error": None,
            }
        except Exception as exc:
            entry["describe"] = {
                "seconds": round(time.time() - t0, 2),
                "chars": 0,
                "prompt": "",
                "error": str(exc),
            }
        print(
            f"[describe] {entry['describe']['seconds']}s chars={entry['describe']['chars']} "
            f"err={entry['describe']['error']}"
        )
        print((entry["describe"]["prompt"] or "")[:450])

        t0 = time.time()
        try:
            text = _run_wand(model, wand_prompt)
            entry["wand"] = {
                "seconds": round(time.time() - t0, 2),
                "chars": len(text or ""),
                "expanded": text,
                "error": None,
            }
        except Exception as exc:
            entry["wand"] = {
                "seconds": round(time.time() - t0, 2),
                "chars": 0,
                "expanded": "",
                "error": str(exc),
            }
        print(
            f"[wand] {entry['wand']['seconds']}s chars={entry['wand']['chars']} "
            f"err={entry['wand']['error']}"
        )
        print((entry["wand"]["expanded"] or "")[:450])
        report["models"][model] = entry

    # Summary
    m4 = report["models"][MODELS[0]]
    m2 = report["models"][MODELS[1]]
    summary = {}
    for task in ("describe", "wand"):
        s4 = m4[task]["seconds"]
        s2 = m2[task]["seconds"]
        summary[task] = {
            "4b_s": s4,
            "2b_s": s2,
            "speedup_x": round(s4 / s2, 2) if s2 else None,
            "4b_ok": not m4[task]["error"] and bool(m4[task].get("prompt") or m4[task].get("expanded")),
            "2b_ok": not m2[task]["error"] and bool(m2[task].get("prompt") or m2[task].get("expanded")),
        }
    report["summary"] = summary
    out = ROOT / "logs" / "ab_abliterated_2b_vs_4b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(json.dumps(summary, indent=2))
    if not all(summary[t]["2b_ok"] and summary[t]["4b_ok"] for t in summary):
        raise SystemExit("A/B FAILED")
    print("A/B OK")


if __name__ == "__main__":
    main()
