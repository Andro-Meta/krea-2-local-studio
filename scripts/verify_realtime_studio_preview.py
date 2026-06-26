from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / f"verify_realtime_studio_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASES = ["http://127.0.0.1:8200/krea", "http://127.0.0.1:8200"]
TURBO_PATH = Path(
    os.environ.get(
        "KREA2_TURBO_PATH",
        str(ROOT / "models" / "krea2" / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors"),
    )
)


def encode(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_b64(name: str, image_b64: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    path.write_bytes(base64.b64decode(image_b64))
    return path


def request_json(base: str, path: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_base() -> str:
    last_error: Exception | None = None
    for base in BASES:
        try:
            request_json(base, "/api/system", timeout=3)
            return base
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Krea server is not reachable without auth at {BASES}: {last_error}")


def wait_model(base: str) -> None:
    last = {}
    for _ in range(240):
        last = request_json(base, "/api/system")
        status = last.get("model_status", {})
        if status.get("loaded"):
            print("MODEL_READY", status, flush=True)
            return
        if status.get("load_error"):
            raise RuntimeError(f"model load failed: {status.get('load_error')}")
        time.sleep(2)
    raise TimeoutError(f"model did not become ready: {last.get('model_status')}")


def ensure_model(base: str) -> None:
    status = request_json(base, "/api/system").get("model_status", {})
    if status.get("loaded"):
        print("model already loaded", flush=True)
        return
    if status.get("loading"):
        wait_model(base)
        return
    if not TURBO_PATH.exists():
        raise FileNotFoundError(f"Krea turbo checkpoint not found: {TURBO_PATH}")
    request_json(base, "/api/load-model", method="POST", payload={"checkpoint_path": str(TURBO_PATH), "quantization": "fp8"})
    wait_model(base)


def wait_preview(base: str, job_id: str) -> dict:
    last = {}
    for _ in range(240):
        last = request_json(base, f"/api/realtime/preview/{job_id}")
        if last.get("status") in {"done", "error", "cancelled", "stale"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"preview did not finish: {last}")


def wait_generation(base: str, job_id: str) -> dict:
    last = {}
    for _ in range(360):
        last = request_json(base, f"/api/generate/{job_id}")
        if last.get("status") in {"done", "error"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"final render did not finish: {last}")


def sketch_canvas() -> str:
    img = Image.new("RGB", (512, 512), (237, 232, 221))
    draw = ImageDraw.Draw(img)
    draw.line((60, 382, 212, 240, 450, 382), fill=(45, 40, 36), width=13, joint="curve")
    draw.rectangle((150, 250, 360, 390), outline=(68, 55, 44), width=10)
    draw.ellipse((342, 68, 430, 156), fill=(230, 176, 70))
    return encode(img)


def asset_canvas() -> str:
    img = Image.new("RGB", (512, 512), (212, 224, 232))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 328, 512, 512), fill=(90, 128, 76))
    draw.rounded_rectangle((176, 120, 338, 310), radius=28, fill=(36, 74, 160))
    draw.rectangle((220, 205, 294, 276), fill=(221, 178, 82))
    return encode(img)


def shape_canvas() -> str:
    img = Image.new("RGB", (512, 512), (194, 188, 178))
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 110, 300, 330), fill=(90, 66, 154))
    draw.rectangle((240, 240, 430, 402), fill=(195, 91, 58))
    draw.polygon([(358, 72), (462, 248), (260, 248)], fill=(222, 170, 74))
    return encode(img)


def submit_preview(base: str, name: str, canvas_b64: str, prompt: str) -> str:
    job = request_json(
        base,
        "/api/realtime/preview",
        method="POST",
        payload={
            "session_id": f"verify-{name}",
            "prompt": prompt,
            "negative_prompt": "low quality, blurry, text, watermark",
            "canvas_image_b64": canvas_b64,
            "width": 512,
            "height": 512,
            "preview_steps": 5,
            "moodboard_strength": 0.6,
        },
    )
    result = wait_preview(base, job["job_id"])
    if result.get("status") != "done" or not result.get("image_b64"):
        raise RuntimeError(f"{name} preview failed: {result}")
    path = save_b64(f"{name}_preview", result["image_b64"])
    print(f"{name}: {path}", flush=True)
    return result["image_b64"]


def submit_final(base: str, canvas_b64: str) -> None:
    job = request_json(
        base,
        "/api/generate",
        method="POST",
        payload={
            "prompt": "photorealistic cozy cabin from the rough realtime canvas sketch, cohesive lighting",
            "negative_prompt": "low quality, blurry, text, watermark",
            "mode": "redraw",
            "checkpoint": "turbo",
            "quantization": "fp8",
            "steps": 8,
            "cfg": 0,
            "width": 512,
            "height": 512,
            "num_images": 1,
            "denoise": 1,
            "moodboard_strength": 0.6,
            "moodboard_images": [canvas_b64],
        },
    )
    result = wait_generation(base, job["job_id"])
    if result.get("status") != "done" or not result.get("images"):
        raise RuntimeError(f"final render failed: {result}")
    path = save_b64("final_render", result["images"][0])
    print(f"final: {path}", flush=True)


def main() -> None:
    base = find_base()
    print(f"using {base}", flush=True)
    ensure_model(base)
    sketch = sketch_canvas()
    submit_preview(base, "sketch", sketch, "photorealistic cozy cabin from a rough sketch")
    submit_preview(base, "asset", asset_canvas(), "photorealistic blue travel backpack in a meadow")
    submit_preview(base, "shape", shape_canvas(), "modern surreal product photograph based on simple colored shapes")
    submit_final(base, sketch)
    print(f"REALTIME_VERIFY_DONE {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
