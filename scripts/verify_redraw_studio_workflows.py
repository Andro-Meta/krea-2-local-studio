from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / f"verify_redraw_studio_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def save_b64(name: str, image_b64: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    path.write_bytes(base64.b64decode(image_b64))
    return str(path)


def request_json(base: str, path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_base() -> str:
    last_error: Exception | None = None
    for base in BASES:
        try:
            request_json(base, "/api/system")
            return base
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Krea server is not reachable without auth at {BASES}: {last_error}")


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
    print(f"loading model: {TURBO_PATH}", flush=True)
    request_json(
        base,
        "/api/load-model",
        method="POST",
        payload={"checkpoint_path": str(TURBO_PATH), "quantization": "fp8"},
    )
    wait_model(base)


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


def wait_job(base: str, job_id: str) -> dict:
    last = {}
    for _ in range(360):
        last = request_json(base, f"/api/generate/{job_id}")
        if last.get("status") in {"done", "error"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"job did not finish: {last}")


def make_scene() -> str:
    img = Image.new("RGB", (512, 512), (219, 231, 226))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 355, 512, 512), fill=(84, 126, 84))
    draw.polygon([(70, 240), (210, 122), (350, 240)], fill=(105, 55, 45))
    draw.rectangle((105, 240, 318, 390), fill=(142, 94, 58))
    draw.rectangle((188, 304, 238, 390), fill=(55, 42, 35))
    draw.rectangle((125, 268, 176, 318), fill=(65, 91, 112))
    draw.rectangle((248, 268, 300, 318), fill=(65, 91, 112))
    draw.ellipse((365, 52, 430, 117), fill=(236, 176, 79))
    draw.line((20, 512, 206, 390), fill=(150, 128, 82), width=14)
    return encode(img)


def make_person() -> str:
    img = Image.new("RGB", (512, 512), (225, 222, 215))
    draw = ImageDraw.Draw(img)
    draw.ellipse((210, 88, 302, 180), fill=(176, 125, 89))
    draw.rectangle((196, 180, 318, 358), fill=(156, 35, 40))
    draw.line((196, 210, 118, 306), fill=(176, 125, 89), width=18)
    draw.line((318, 210, 392, 302), fill=(176, 125, 89), width=18)
    draw.line((226, 358, 205, 468), fill=(42, 56, 78), width=24)
    draw.line((288, 358, 310, 468), fill=(42, 56, 78), width=24)
    return encode(img)


def make_object() -> str:
    img = Image.new("RGB", (512, 512), (230, 229, 224))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((150, 160, 362, 376), radius=40, fill=(35, 72, 145))
    draw.arc((185, 90, 327, 230), 180, 360, fill=(35, 72, 145), width=22)
    draw.rectangle((190, 260, 322, 332), fill=(215, 167, 65))
    return encode(img)


def make_sketch() -> str:
    img = Image.new("RGB", (512, 512), (245, 243, 237))
    draw = ImageDraw.Draw(img)
    draw.rectangle((120, 235, 390, 390), outline=(50, 45, 42), width=8)
    draw.polygon([(90, 235), (256, 112), (422, 235)], outline=(88, 42, 40), width=10)
    draw.line((28, 512, 216, 390), fill=(95, 82, 65), width=7)
    draw.line((0, 410, 512, 410), fill=(84, 110, 74), width=6)
    draw.ellipse((366, 54, 428, 116), outline=(210, 145, 63), width=7)
    return encode(img)


def make_style() -> str:
    img = Image.new("RGB", (512, 512), (28, 32, 43))
    draw = ImageDraw.Draw(img)
    for radius, color in [(240, (26, 36, 65)), (180, (58, 76, 126)), (120, (200, 143, 76))]:
        draw.ellipse((256 - radius, 256 - radius, 256 + radius, 256 + radius), fill=color)
    img = img.filter(ImageFilter.GaussianBlur(radius=18))
    return encode(img)


def make_mood() -> str:
    img = Image.new("RGB", (512, 512), (42, 50, 72))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 270, 512, 512), fill=(16, 24, 36))
    draw.ellipse((360, 42, 450, 132), fill=(210, 218, 232))
    for x in range(20, 512, 72):
        draw.line((x, 265, x - 42, 512), fill=(62, 86, 72), width=18)
    return encode(img)


def make_inpaint_mask() -> str:
    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((206, 160, 318, 288), fill=255)
    return encode(mask)


def make_outpaint_source() -> tuple[str, str, int, int]:
    src = Image.open(io.BytesIO(base64.b64decode(make_scene()))).convert("RGB").resize((512, 512))
    expanded = Image.new("RGB", (912, 512), (128, 128, 128))
    expanded.paste(src, (200, 0))
    mask = Image.new("L", expanded.size, 255)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle((200, 0, 711, 511), fill=0)
    for x in range(128):
        value = int(255 * ((128 - x) / 128) ** 2)
        draw_mask.line((200 + x, 0, 200 + x, 511), fill=value)
        draw_mask.line((711 - x, 0, 711 - x, 511), fill=value)
    return encode(expanded), encode(mask), expanded.width, expanded.height


def common_payload() -> dict:
    return {
        "checkpoint": "turbo",
        "quantization": "fp8",
        "width": 512,
        "height": 512,
        "steps": 6,
        "cfg": 0,
        "num_images": 1,
        "seed": 260626,
        "denoise": 1.0,
        "use_rebalance": True,
        "rebalance_multiplier": 4.0,
        "rebalance_weights": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
        "loras": [],
        "bboxes": [],
    }


def workflow_payloads(refs: dict[str, str]) -> dict[str, dict]:
    outpaint_init, outpaint_mask, outpaint_w, outpaint_h = make_outpaint_source()
    base = common_payload()
    return {
        "recreate_redraw": {
            **base,
            "mode": "redraw",
            "prompt": "Picture 1 is the scene/location and composition. Create a finished realistic cozy cabin photograph from the reference.",
            "moodboard_images": [refs["scene"]],
            "moodboard_strength": 0.75,
        },
        "add_or_replace": {
            **base,
            "mode": "redraw",
            "prompt": "Picture 1 is the scene. Picture 2 is the person. Picture 3 is the blue backpack. Place the person and backpack into the scene with coherent lighting and scale.",
            "moodboard_images": [refs["scene"], refs["person"], refs["object"]],
            "moodboard_strength": 0.75,
        },
        "sketch_to_realistic": {
            **base,
            "mode": "redraw",
            "prompt": "Picture 1 is the sketch/layout reference. Render it as a realistic golden-hour cabin photograph with coherent materials and lighting.",
            "moodboard_images": [refs["sketch"]],
            "moodboard_strength": 0.75,
        },
        "style_transfer": {
            **base,
            "mode": "redraw",
            "prompt": "Picture 1 is the subject and composition. Picture 2 is the visual style. Redraw the cabin using the cinematic blue-orange style reference.",
            "moodboard_images": [refs["scene"], refs["style"]],
            "moodboard_strength": 0.8,
        },
        "moodboard_direction": {
            **base,
            "mode": "redraw",
            "prompt": "Create a new atmospheric cabin scene using the shared color, lighting, and mood from these references.",
            "moodboard_images": [refs["mood"], refs["style"], refs["scene"]],
            "moodboard_strength": 0.85,
        },
        "extend_creative_redraw": {
            **base,
            "mode": "redraw",
            "prompt": "Picture 1 is the source image to extend. Redraw it into a seamless 16:9 wide cinematic cabin landscape.",
            "width": 912,
            "height": 512,
            "moodboard_images": [refs["scene"]],
            "moodboard_strength": 0.75,
        },
        "extend_preserve_outpaint": {
            **base,
            "mode": "outpaint",
            "prompt": "Extend this cabin scene into a wider landscape with no visible seams, matching lighting and perspective.",
            "width": outpaint_w,
            "height": outpaint_h,
            "init_image_b64": outpaint_init,
            "mask_b64": outpaint_mask,
            "moodboard_images": [],
        },
        "preserve_whole_img2img": {
            **base,
            "mode": "img2img",
            "prompt": "Enhance this cabin image into a more detailed realistic photograph while preserving composition.",
            "init_image_b64": refs["scene"],
            "denoise": 0.45,
            "moodboard_images": [],
        },
        "preserve_masked_inpaint": {
            **base,
            "mode": "inpaint",
            "prompt": "Add a warm glowing round lantern in the masked area, matching the scene lighting.",
            "init_image_b64": refs["scene"],
            "mask_b64": refs["mask"],
            "denoise": 0.75,
            "moodboard_images": [refs["object"]],
            "moodboard_strength": 0.55,
        },
    }


def generate(base: str, label: str, payload: dict) -> dict:
    print(f"starting {label}", flush=True)
    job = request_json(base, "/api/generate", method="POST", payload=payload)
    result = wait_job(base, job["job_id"])
    if result.get("status") != "done" or not result.get("images"):
        raise RuntimeError(f"{label} failed: {result.get('error') or result}")
    output_path = save_b64(label, result["images"][0])
    print(f"done {label}: seed={result.get('seed')} file={output_path}", flush=True)
    return {"seed": result.get("seed"), "file": output_path, "image_count": len(result.get("images", []))}


def main() -> None:
    base = find_base()
    ensure_model(base)
    refs = {
        "scene": make_scene(),
        "person": make_person(),
        "object": make_object(),
        "sketch": make_sketch(),
        "style": make_style(),
        "mood": make_mood(),
        "mask": make_inpaint_mask(),
    }
    for name, b64 in refs.items():
        save_b64(f"ref_{name}", b64)

    summary = {}
    for label, payload in workflow_payloads(refs).items():
        summary[label] = generate(base, label, payload)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "workflows": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
