from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


def _json_request(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"content-type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _png_b64(img: Image.Image) -> str:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fixture_images(out_dir: Path) -> tuple[str, str, str]:
    source = Image.new("RGB", (512, 512), "#2b2b34")
    draw = ImageDraw.Draw(source)
    draw.rectangle((120, 140, 390, 390), fill="#d8c7a3")
    draw.ellipse((185, 185, 325, 325), fill="#2d5a88")
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).ellipse((185, 185, 325, 325), fill=255)
    style = Image.new("RGB", (512, 512), "#101018")
    ImageDraw.Draw(style).line((0, 512, 512, 0), fill="#ff77aa", width=24)
    for name, img in {"source.png": source, "mask.png": mask, "style.png": style}.items():
        img.save(out_dir / name)
    return _png_b64(source), _png_b64(mask.convert("RGB")), _png_b64(style)


def _poll_job(base_url: str, job_id: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = _json_request(f"{base_url}/api/generate/{job_id}", timeout=10)
        if job.get("status") in {"done", "error"}:
            return job
        time.sleep(2)
    return {"status": "timeout", "job_id": job_id}


def _save_images(job: dict, case_dir: Path) -> None:
    for idx, image_b64 in enumerate(job.get("images") or []):
        data = base64.b64decode(image_b64)
        (case_dir / f"output_{idx + 1}.png").write_bytes(data)
    (case_dir / "metadata.json").write_text(json.dumps(job.get("metadata") or [], indent=2), encoding="utf-8")


def _contact_sheet(out_dir: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in sorted(out_dir.glob("*/output_*.png"))]
    if not images:
        return
    thumb_w = 256
    thumbs = [img.resize((thumb_w, int(img.height * thumb_w / img.width))) for img in images]
    sheet = Image.new("RGB", (thumb_w * len(thumbs), max(img.height for img in thumbs)), "white")
    x = 0
    for img in thumbs:
        sheet.paste(img, (x, 0))
        x += thumb_w
    sheet.save(out_dir / "contact_sheet.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ComfyUI Krea 2 parity controls against the local API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8200")
    parser.add_argument("--out-dir", default="new_inpaint_tests/comfy_parity")
    parser.add_argument("--run-generation", action="store_true", help="Run live generation cases instead of API/status only.")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review: list[str] = ["# Comfy Krea Parity Visual Review", ""]

    try:
        system = _json_request(f"{args.base_url}/api/system", timeout=10)
        (out_dir / "system.json").write_text(json.dumps(system, indent=2), encoding="utf-8")
        review.append("- API system check: reachable.")
    except URLError as exc:
        review.append(f"- API system check: unavailable ({exc}).")
        (out_dir / "VISUAL_REVIEW.md").write_text("\n".join(review) + "\n", encoding="utf-8")
        return 1

    source_b64, mask_b64, style_b64 = _fixture_images(out_dir)
    cases = [
        ("txt2img_turbo_default", {"prompt": "a cinematic glass apple on a marble table", "seed": 101}),
        ("txt2img_style_ref", {
            "prompt": "a futuristic perfume bottle, studio product photo",
            "seed": 102,
            "style_references": [{"image_b64": style_b64, "strength": 1.0, "token_size": "normal"}],
        }),
        ("inpaint_native_default", {
            "prompt": "replace the blue circle with a small brass lantern",
            "mode": "inpaint",
            "init_image_b64": source_b64,
            "mask_b64": mask_b64,
            "denoise": 1.0,
            "seed": 103,
        }),
        ("seed_variance_ab", {
            "prompt": "a misty forest shrine at sunrise",
            "seed": 104,
            "seed_variance_preset": "balanced",
        }),
    ]

    if args.run_generation:
        for name, payload in cases:
            case_dir = out_dir / name
            case_dir.mkdir(exist_ok=True)
            (case_dir / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            try:
                started = _json_request(f"{args.base_url}/api/generate", payload, timeout=30)
                job = _poll_job(args.base_url, started["job_id"], args.timeout_seconds)
                (case_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
                _save_images(job, case_dir)
                review.append(f"- {name}: {job.get('status')}.")
            except Exception as exc:  # noqa: BLE001
                review.append(f"- {name}: failed ({exc}).")
        _contact_sheet(out_dir)
    else:
        review.append("- Live generation skipped. Re-run with `--run-generation` after loading Turbo FP8.")

    review.append("")
    review.append("Manual checks: inspect any outputs/contact sheet and note whether style refs, inpaint, moodboard UUIDs, edit rebalance, and seed variance improve or regress quality.")
    (out_dir / "VISUAL_REVIEW.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
