from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_image(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def request_json(base_url: str, path: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local test URL
        return json.loads(resp.read().decode("utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def wait_job(base_url: str, job_id: str, *, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        last = request_json(base_url, f"/api/generate/{job_id}", timeout=30)
        if last.get("status") in {"done", "error"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"job did not finish: {job_id}; last={last}")


def wait_preview(base_url: str, job_id: str, *, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        last = request_json(base_url, f"/api/realtime/preview/{job_id}", timeout=30)
        if last.get("status") in {"done", "error", "cancelled", "stale"}:
            return last
        time.sleep(2)
    raise TimeoutError(f"preview did not finish: {job_id}; last={last}")


def ensure_model(base_url: str, timeout_s: int) -> dict:
    status = request_json(base_url, "/api/system", timeout=30)
    model_status = status.get("model_status", {})
    if model_status.get("loaded"):
        return status
    if not model_status.get("loading"):
        turbo_path = ROOT / "models" / "krea2" / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors"
        request_json(
            base_url,
            "/api/load-model",
            method="POST",
            payload={"checkpoint_path": str(turbo_path), "quantization": "fp8"},
            timeout=60,
        )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = request_json(base_url, "/api/system", timeout=30)
        model_status = status.get("model_status", {})
        if model_status.get("loaded"):
            return status
        if model_status.get("load_error"):
            raise RuntimeError(f"model load failed: {model_status.get('load_error')}")
        time.sleep(3)
    raise TimeoutError("model did not finish loading")


def make_source() -> Image.Image:
    image = Image.new("RGB", (512, 512), (222, 224, 216))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 355, 512, 512), fill=(90, 116, 75))
    draw.rectangle((96, 190, 390, 370), fill=(148, 92, 55))
    draw.polygon([(72, 190), (244, 72), (416, 190)], fill=(86, 48, 42))
    draw.rectangle((205, 275, 265, 370), fill=(48, 36, 28))
    draw.ellipse((372, 48, 442, 118), fill=(232, 180, 80))
    return image


def make_style_ref() -> Image.Image:
    image = Image.new("RGB", (512, 512), (32, 34, 48))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 330, 512, 512), fill=(18, 24, 36))
    draw.ellipse((358, 40, 456, 138), fill=(210, 220, 236))
    for x in range(40, 512, 90):
        draw.line((x, 330, x - 62, 512), fill=(58, 88, 72), width=18)
    return image


def make_mask() -> Image.Image:
    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((170, 120, 330, 280), fill=255)
    return mask


def make_outpaint() -> tuple[Image.Image, Image.Image]:
    src = make_source().resize((384, 384))
    expanded = Image.new("RGB", (640, 384), (72, 80, 88))
    expanded.paste(src, (128, 0))
    mask = Image.new("L", expanded.size, 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((128, 0, 511, 383), fill=0)
    draw.rectangle((128, 0, 160, 383), fill=255)
    draw.rectangle((480, 0, 511, 383), fill=255)
    return expanded, mask


def contact_sheet(items: list[tuple[str, Image.Image]], path: Path) -> None:
    thumb_w, thumb_h = 240, 240
    sheet = Image.new("RGB", (thumb_w * len(items), thumb_h + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(items):
        preview = image.convert("RGB")
        preview.thumbnail((thumb_w, thumb_h))
        x = idx * thumb_w + (thumb_w - preview.width) // 2
        y = 30 + (thumb_h - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((idx * thumb_w + 6, 8), label, fill=(0, 0, 0))
    sheet.save(path)


def generation_payloads(source_b64: str, style_b64: str, mask_b64: str, outpaint_b64: str, outpaint_mask_b64: str) -> dict[str, dict]:
    common = {
        "model_profile": "krea_turbo",
        "checkpoint": "turbo",
        "quantization": "fp8",
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 270627,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "negative_prompt": "low quality, blurry, text, watermark, distorted",
    }
    return {
        "txt2img_default": {
            **common,
            "mode": "txt2img",
            "prompt": "a cozy cottage beside a winding path, golden hour, detailed realistic photography",
            "steps": 8,
            "denoise": 1.0,
        },
        "txt2img_heun": {
            **common,
            "mode": "txt2img",
            "prompt": "a translucent blue glass chair in a sunlit design studio, product photography",
            "sampler": "exp_heun_2_x0_sde",
            "steps": 6,
            "denoise": 1.0,
        },
        "txt2img_seed_variance": {
            **common,
            "mode": "txt2img",
            "prompt": "a cinematic portrait of a ceramic robot gardener in a greenhouse",
            "steps": 8,
            "seed_variance_preset": "balanced",
            "seed_variance_strength": 0.35,
        },
        "redraw_reference": {
            **common,
            "mode": "redraw",
            "prompt": "use the simple sketch only for layout and silhouette; create a finished realistic cottage photograph with natural materials",
            "steps": 8,
            "cfg": 0.0,
            "denoise": 0.9,
            "moodboard_images": [source_b64],
            "moodboard_strength": 0.42,
        },
        "img2img_style_ref": {
            **common,
            "mode": "img2img",
            "prompt": "a cozy cottage in a moonlit forest, keep the layout but use a moody cinematic style",
            "steps": 8,
            "denoise": 0.65,
            "init_image_b64": source_b64,
            "style_references": [{"image_b64": style_b64, "strength": 0.8, "role": "style", "token_size": "normal"}],
        },
        "inpaint_native": {
            **common,
            "mode": "inpaint",
            "prompt": "replace the masked area with a glowing round attic window, coherent lighting",
            "steps": 8,
            "denoise": 1.0,
            "init_image_b64": source_b64,
            "mask_b64": mask_b64,
            "inpaint_method": "native",
        },
        "inpaint_lanpaint": {
            **common,
            "mode": "inpaint",
            "prompt": "replace the masked area with a glowing round attic window, coherent lighting",
            "steps": 20,
            "denoise": 1.0,
            "init_image_b64": source_b64,
            "mask_b64": mask_b64,
            "inpaint_method": "lanpaint_experimental",
            "lanpaint_inner_steps": 5,
            "lanpaint_lambda": 16.0,
            "lanpaint_step_size": 0.2,
            "lanpaint_beta": 1.0,
            "lanpaint_friction": 15.0,
            "lanpaint_early_stop": 1,
        },
        "outpaint_native": {
            **common,
            "mode": "outpaint",
            "prompt": "extend the cottage scene into a wider landscape with trees and a winding path",
            "steps": 8,
            "width": 640,
            "height": 384,
            "denoise": 1.0,
            "init_image_b64": outpaint_b64,
            "mask_b64": outpaint_mask_b64,
        },
    }


def run_generation_case(base_url: str, label: str, payload: dict, out_dir: Path, timeout_s: int) -> tuple[str, Image.Image | None, dict]:
    job = request_json(base_url, "/api/generate", method="POST", payload=payload, timeout=60)
    result = wait_job(base_url, job["job_id"], timeout_s=timeout_s)
    save_json(out_dir / f"{label}_job.json", result)
    if result.get("status") != "done":
        return f"{label}: failed: {result.get('error')}", None, result
    if not result.get("images"):
        return f"{label}: failed: no image returned", None, result
    image = decode_image(result["images"][0])
    image.save(out_dir / f"{label}.png")
    return f"{label}: ok seed={result.get('seed')}", image, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Krea Studio full-system smoke and visual tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8200")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "full_system_tests"))
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    out_dir = Path(args.out) / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    api_checks: dict[str, Any] = {}

    system = ensure_model(args.base_url, args.timeout)
    api_checks["system"] = system

    for label, path in {
        "settings": "/api/settings",
        "moods": "/api/moods",
        "gallery": "/api/gallery?limit=5",
        "moodboards": "/api/moodboards?q=cinematic",
        "moodboard_discoveries": "/api/moodboards/discoveries/latest",
    }.items():
        try:
            api_checks[label] = request_json(args.base_url, path, timeout=60)
            notes.append(f"{label}: ok")
        except Exception as exc:
            api_checks[label] = {"error": str(exc)}
            notes.append(f"{label}: failed: {exc}")

    source = make_source()
    style = make_style_ref()
    mask = make_mask()
    outpaint, outpaint_mask = make_outpaint()
    fixtures = {
        "fixture_source": source,
        "fixture_style": style,
        "fixture_mask": mask.convert("RGB"),
        "fixture_outpaint": outpaint,
        "fixture_outpaint_mask": outpaint_mask.convert("RGB"),
    }
    for label, image in fixtures.items():
        image.save(out_dir / f"{label}.png")

    try:
        preview = request_json(
            args.base_url,
            "/api/preprocess/preview",
            method="POST",
            payload={"image_b64": encode_image(source), "kind": "canny", "resolution": 512},
            timeout=120,
        )
        api_checks["preprocessor_preview"] = {k: v for k, v in preview.items() if k != "image_b64"}
        decode_image(preview["image_b64"]).save(out_dir / "preprocessor_canny.png")
        notes.append("preprocessor_preview: ok")
    except Exception as exc:
        api_checks["preprocessor_preview"] = {"error": str(exc)}
        notes.append(f"preprocessor_preview: failed: {exc}")

    moodboard_ids: list[int] = []
    items = api_checks.get("moodboards", {}).get("items", [])
    if items:
        first_id = int(items[0]["id"])
        moodboard_ids = [first_id]
        try:
            detail = request_json(args.base_url, f"/api/moodboards/{first_id}", timeout=60)
            api_checks["moodboard_detail"] = {"id": detail.get("id"), "title": detail.get("title")}
            favorite = request_json(args.base_url, f"/api/moodboards/{first_id}/favorite", method="PUT", payload={"favorite": True}, timeout=60)
            api_checks["moodboard_favorite"] = favorite
            notes.append(f"moodboard_detail/favorite: ok id={first_id}")
        except Exception as exc:
            api_checks["moodboard_detail"] = {"error": str(exc)}
            notes.append(f"moodboard_detail/favorite: failed: {exc}")

    payloads = generation_payloads(
        encode_image(source),
        encode_image(style),
        encode_image(mask),
        encode_image(outpaint),
        encode_image(outpaint_mask),
    )
    if moodboard_ids:
        payloads["txt2img_catalog_moodboard"] = {
            **payloads["txt2img_default"],
            "prompt": "a gritty cinematic street portrait with natural lighting and tactile texture",
            "moodboard_ids": moodboard_ids,
            "moodboard_strength": 0.45,
        }

    generated: list[tuple[str, Image.Image]] = list(fixtures.items())
    for label, payload in payloads.items():
        note, image, result = run_generation_case(args.base_url, label, payload, out_dir, args.timeout)
        notes.append(note)
        metadata = result.get("metadata") or []
        api_checks[f"{label}_metadata"] = metadata[0] if metadata else {}
        if image is not None:
            generated.append((label, image))

    try:
        upscale_source = next((image for label, image in generated if label == "txt2img_default"), generated[-1][1])
        upscaled = request_json(
            args.base_url,
            "/api/upscale",
            method="POST",
            payload={
                "image_b64": encode_image(upscale_source),
                "method": "ultimate",
                "prompt": "refine details, coherent texture",
                "upscale_by": 1.25,
                "tile_width": 512,
                "tile_height": 512,
                "steps": 4,
                "cfg": 1.0,
                "denoise": 0.18,
                "sampler": "euler",
                "scheduler": "simple",
            },
            timeout=args.timeout,
        )
        decode_image(upscaled["image_b64"]).save(out_dir / "upscale_ultimate.png")
        api_checks["upscale_ultimate"] = {k: v for k, v in upscaled.items() if k != "image_b64"}
        notes.append("upscale_ultimate: ok")
    except Exception as exc:
        api_checks["upscale_ultimate"] = {"error": str(exc)}
        notes.append(f"upscale_ultimate: failed: {exc}")

    try:
        preview_job = request_json(
            args.base_url,
            "/api/realtime/preview",
            method="POST",
            payload={
                "session_id": "full-system-test",
                "prompt": "photorealistic cozy cottage from a rough canvas sketch",
                "negative_prompt": "low quality, blurry, text, watermark",
                "canvas_image_b64": encode_image(source),
                "width": 512,
                "height": 512,
                "preview_steps": 5,
                "moodboard_strength": 0.45,
                "seed": 270627,
            },
            timeout=60,
        )
        preview = wait_preview(args.base_url, preview_job["job_id"], timeout_s=args.timeout)
        save_json(out_dir / "realtime_preview_job.json", preview)
        if preview.get("status") == "done" and preview.get("image_b64"):
            decode_image(preview["image_b64"]).save(out_dir / "realtime_preview.png")
            notes.append("realtime_preview: ok")
        else:
            notes.append(f"realtime_preview: failed: {preview.get('error')}")
    except Exception as exc:
        notes.append(f"realtime_preview: failed: {exc}")

    contact_sheet(generated, out_dir / "contact_sheet.png")
    save_json(out_dir / "api_checks.json", api_checks)

    review = [
        "# Full System Test",
        "",
        "## Scope",
        "- Local-only API/system checks.",
        "- txt2img defaults and sampler variants.",
        "- redraw/img2img/style references.",
        "- native inpaint, LanPaint inpaint, outpaint.",
        "- moodboard search/detail/favorite and catalog moodboard generation when local catalog data is present.",
        "- preprocessor preview, realtime preview, gallery surface, and Ultimate upscale.",
        "",
        "## Results",
        *[f"- {note}" for note in notes],
        "",
        "## Visual Review Checklist",
        "- Open `contact_sheet.png` and individual PNGs.",
        "- Confirm prompt alignment, coherent anatomy/objects, and no severe artifacts.",
        "- Compare native vs LanPaint boundaries.",
        "- Check outpaint edge continuity.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(review), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
