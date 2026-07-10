"""A/B Studio Transformers Qwen vs ComfyUI-QwenVL for describe + magic wand.

Usage (from repo root, Studio venv):
  venv\\Scripts\\python.exe scripts\\ab_qwen_helper.py [image_path]
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from prompt_expander import (  # noqa: E402
    EXPANSION_SYSTEM_PROMPT,
    _describe_prompt,
    describe_image_comfy,
    describe_image_transformers,
    expand_prompt_comfy,
    expand_prompt_transformers,
)
from comfy_qwen_vl import comfy_qwen_vl_available  # noqa: E402


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
    raise SystemExit("No sample image found under outputs/ or data/custom_moodboards/")


def main() -> None:
    img_path = _find_image(sys.argv[1] if len(sys.argv) > 1 else None)
    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    wand_prompt = "a lonely lighthouse on a cliff at dusk"
    describe_mode = "recreate"
    report: dict = {
        "image": str(img_path),
        "comfy_qwenvl_available": comfy_qwen_vl_available(),
        "describe_instruction": _describe_prompt(describe_mode),
        "wand_prompt": wand_prompt,
    }
    print(f"image={img_path}")
    print(f"comfy_qwenvl_available={report['comfy_qwenvl_available']}")

    # --- describe ---
    t0 = time.time()
    studio = describe_image_transformers(b64, describe_mode)
    report["describe_studio"] = {
        "seconds": round(time.time() - t0, 2),
        "backend": studio.get("backend"),
        "prompt": studio.get("prompt", ""),
        "chars": len(studio.get("prompt") or ""),
    }
    print(f"\n[describe studio] {report['describe_studio']['seconds']}s chars={report['describe_studio']['chars']}")
    print(report["describe_studio"]["prompt"][:500])

    t0 = time.time()
    comfy = describe_image_comfy(b64, describe_mode)
    report["describe_comfy"] = {
        "seconds": round(time.time() - t0, 2),
        "backend": comfy.get("backend"),
        "prompt": comfy.get("prompt", ""),
        "chars": len(comfy.get("prompt") or ""),
    }
    print(f"\n[describe comfy] {report['describe_comfy']['seconds']}s chars={report['describe_comfy']['chars']}")
    print(report["describe_comfy"]["prompt"][:500])

    # --- magic wand ---
    t0 = time.time()
    wand_s = expand_prompt_transformers(wand_prompt)
    report["wand_studio"] = {
        "seconds": round(time.time() - t0, 2),
        "backend": wand_s.backend,
        "expanded": wand_s.expanded,
        "chars": len(wand_s.expanded or ""),
        "error": wand_s.error,
    }
    print(f"\n[wand studio] {report['wand_studio']['seconds']}s chars={report['wand_studio']['chars']}")
    print((wand_s.expanded or "")[:500])

    t0 = time.time()
    wand_c = expand_prompt_comfy(wand_prompt)
    report["wand_comfy"] = {
        "seconds": round(time.time() - t0, 2),
        "backend": wand_c.backend,
        "expanded": wand_c.expanded,
        "chars": len(wand_c.expanded or ""),
        "error": wand_c.error,
    }
    print(f"\n[wand comfy] {report['wand_comfy']['seconds']}s chars={report['wand_comfy']['chars']}")
    print((wand_c.expanded or "")[:500])

    out = ROOT / "logs" / "ab_qwen_helper.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    ok = bool(report["describe_comfy"]["prompt"]) and bool(report["wand_comfy"]["expanded"]) and not report["wand_comfy"]["error"]
    if not ok:
        raise SystemExit("A/B FAILED: Comfy path returned empty/error output")
    print("A/B OK: Comfy path produced non-empty describe + wand outputs")


if __name__ == "__main__":
    main()
