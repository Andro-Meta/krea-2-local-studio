from __future__ import annotations

import base64
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from share_auth import add_user, remove_user  # noqa: E402

BASE = "http://127.0.0.1:8200/krea"
USER = f"krea_verify_{int(time.time())}"
PASSWORD = "VerifyPass-2026"
AUTH_FILE = ROOT / "share_auth.json"
RUN_DIR = ROOT / "outputs" / f"verify_moodboard_outpaint_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

SAMPLES = {
    "photo": "https://picsum.photos/seed/krea-verify-photo/768/768",
    "painting": "https://commons.wikimedia.org/wiki/Special:FilePath/Van%20Gogh%20-%20Starry%20Night%20-%20Google%20Art%20Project.jpg?width=640",
    "print": "https://commons.wikimedia.org/wiki/Special:FilePath/The%20Great%20Wave%20off%20Kanagawa.jpg?width=640",
}


def request_json(path: str, *, method: str = "GET", payload: dict | None = None, cookie: str = "") -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def login() -> str:
    data = json.dumps({"username": USER, "password": PASSWORD}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/auth/login",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        cookies = resp.headers.get_all("Set-Cookie") or []
    session = [c.split(";", 1)[0] for c in cookies if c.startswith("krea_share_session=")]
    if not session:
        raise RuntimeError("login did not return krea_share_session")
    return session[0]


def fetch_b64(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Krea2Verification/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return base64.b64encode(resp.read()).decode("ascii")


def make_outpaint_source() -> tuple[str, str]:
    src = Image.new("RGB", (384, 384), (28, 32, 43))
    draw = ImageDraw.Draw(src)
    draw.rectangle((70, 120, 314, 300), fill=(132, 94, 60))
    draw.polygon([(70, 120), (192, 45), (314, 120)], fill=(80, 42, 39))
    draw.rectangle((165, 210, 220, 300), fill=(48, 35, 30))
    draw.ellipse((36, 36, 96, 96), fill=(238, 207, 141))
    expanded = Image.new("RGB", (640, 384), (18, 20, 28))
    expanded.paste(src, (128, 0))
    mask = Image.new("L", (640, 384), 255)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle((128, 0, 511, 383), fill=0)
    draw_mask.rectangle((128, 0, 160, 383), fill=255)
    draw_mask.rectangle((480, 0, 511, 383), fill=255)
    import io

    def encode(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return encode(expanded), encode(mask)


def make_side_outpaint_from_b64(source_b64: str, pad: int = 128, overlap: int = 32) -> tuple[str, str, int, int]:
    import io

    src = Image.open(io.BytesIO(base64.b64decode(source_b64))).convert("RGB")
    width, height = src.size
    expanded = Image.new("RGB", (width + pad * 2, height), (18, 20, 28))
    expanded.paste(src, (pad, 0))
    mask = Image.new("L", expanded.size, 255)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle((pad, 0, pad + width - 1, height - 1), fill=0)
    draw_mask.rectangle((pad, 0, pad + overlap, height - 1), fill=255)
    draw_mask.rectangle((pad + width - overlap, 0, pad + width - 1, height - 1), fill=255)

    def encode(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return encode(expanded), encode(mask), expanded.width, expanded.height


def wait_job(job_id: str, cookie: str) -> dict:
    last = {}
    for _ in range(240):
        last = request_json(f"/api/generate/{job_id}", cookie=cookie)
        if last.get("status") in {"done", "error"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish; last={last}")


def generate(cookie: str, label: str, payload: dict) -> dict:
    print(f"starting {label}", flush=True)
    job = request_json("/api/generate", method="POST", payload=payload, cookie=cookie)
    result = wait_job(job["job_id"], cookie)
    if result.get("status") != "done":
        raise RuntimeError(f"{label} failed: {result.get('error')}")
    print(f"done {label}: seed={result.get('seed')} images={len(result.get('images', []))}", flush=True)
    return result


def wait_model(cookie: str) -> dict:
    last = {}
    for _ in range(120):
        last = request_json("/api/system", cookie=cookie)
        status = last.get("model_status", {})
        if status.get("loaded"):
            return last
        if status.get("load_error"):
            raise RuntimeError(f"model load failed: {status.get('load_error')}")
        time.sleep(2)
    raise TimeoutError(f"model did not become ready; last={last.get('model_status')}")


def describe(cookie: str, image_b64: str) -> str:
    try:
        return request_json("/api/describe-image", method="POST", payload={"image_b64": image_b64}, cookie=cookie)["prompt"]
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")[:250]
        except Exception:
            body = str(exc)
        return f"describe failed: HTTP {exc.code} {body}"
    except Exception as exc:
        return f"describe failed: {exc}"


def save_b64(name: str, image_b64: str) -> str:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / f"{name}.png"
    path.write_bytes(base64.b64decode(image_b64))
    return str(path)


def main() -> None:
    add_user(AUTH_FILE, USER, PASSWORD)
    try:
        cookie = login()
        system = wait_model(cookie)
        print("model loaded:", system.get("model_status", {}).get("loaded"), flush=True)
        refs = {name: fetch_b64(url) for name, url in SAMPLES.items()}
        print("downloaded refs:", ", ".join(f"{k}:{len(v)}" for k, v in refs.items()), flush=True)
        for name, b64 in refs.items():
            save_b64(f"ref_{name}", b64)

        common = {
            "checkpoint": "turbo",
            "quantization": "fp8",
            "width": 512,
            "height": 512,
            "steps": 6,
            "cfg": 0,
            "num_images": 1,
            "seed": 260626,
            "denoise": 0.9,
            "use_rebalance": True,
            "rebalance_multiplier": 4.0,
            "rebalance_weights": "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0",
            "loras": [],
            "bboxes": [],
        }
        jobs = {}
        jobs["baseline"] = generate(cookie, "baseline", {
            **common,
            "prompt": "a cozy cottage beside a winding path, centered composition",
            "mode": "txt2img",
            "moodboard_strength": 0.5,
            "moodboard_images": [],
        })
        jobs["photo_ref"] = generate(cookie, "photo moodboard", {
            **common,
            "prompt": "a cozy cottage beside a winding path, centered composition",
            "mode": "txt2img",
            "moodboard_strength": 1.0,
            "moodboard_images": [refs["photo"]],
        })
        jobs["painting_ref"] = generate(cookie, "painting moodboard", {
            **common,
            "prompt": "a cozy cottage beside a winding path, centered composition",
            "mode": "txt2img",
            "moodboard_strength": 1.0,
            "moodboard_images": [refs["painting"]],
        })
        init_b64, mask_b64 = make_outpaint_source()
        jobs["outpaint"] = generate(cookie, "outpaint", {
            **common,
            "prompt": "extend the moonlit cottage scene into a wider cinematic landscape, matching lighting and perspective",
            "mode": "outpaint",
            "width": 640,
            "height": 384,
            "steps": 6,
            "denoise": 1.0,
            "init_image_b64": init_b64,
            "mask_b64": mask_b64,
            "moodboard_images": [],
        })
        photo_init, photo_mask, photo_w, photo_h = make_side_outpaint_from_b64(jobs["photo_ref"]["images"][0])
        jobs["outpaint_photo_ref"] = generate(cookie, "outpaint photoreal source", {
            **common,
            "prompt": "extend this golden-hour riverside cabin photograph into a wider landscape, matching the same sunlight, lens, trees, river, mountains, color grading, and perspective",
            "mode": "outpaint",
            "width": photo_w,
            "height": photo_h,
            "steps": 8,
            "denoise": 1.0,
            "init_image_b64": photo_init,
            "mask_b64": photo_mask,
            "moodboard_images": [],
        })

        summary = {}
        for label, job in jobs.items():
            image = job["images"][0]
            path = save_b64(label, image)
            summary[label] = {
                "seed": job.get("seed"),
                "file": path,
                "description": describe(cookie, image),
            }
        (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        remove_user(AUTH_FILE, USER)


if __name__ == "__main__":
    main()
