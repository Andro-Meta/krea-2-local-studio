from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def _b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_json(base_url: str, path: str, payload: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local/user supplied URL
        return json.loads(response.read().decode("utf-8"))


def _get_json(base_url: str, path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _source_and_mask() -> tuple[Image.Image, Image.Image]:
    source = Image.new("RGB", (512, 512), (224, 220, 205))
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 360, 512, 512), fill=(120, 96, 72))
    draw.rectangle((80, 88, 432, 370), fill=(185, 176, 150))
    draw.ellipse((160, 130, 352, 322), fill=(150, 40, 38))
    draw.rectangle((232, 245, 280, 355), fill=(100, 62, 40))
    draw.text((170, 65), "Replace the red object", fill=(40, 40, 40))

    mask = Image.new("L", source.size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse((150, 120, 362, 332), fill=255)
    return source, mask


def _save_contact_sheet(items: list[tuple[str, Image.Image]], path: Path) -> None:
    thumb = 256
    sheet = Image.new("RGB", (thumb * len(items), thumb + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(items):
        preview = image.convert("RGB")
        preview.thumbnail((thumb, thumb))
        x = idx * thumb + (thumb - preview.width) // 2
        y = 28 + (thumb - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((idx * thumb + 6, 8), label, fill=(0, 0, 0))
    sheet.save(path)


def _run_case(base_url: str, payload: dict, timeout_s: int) -> dict:
    job = _post_json(base_url, "/api/generate", payload)
    job_id = job["job_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = _get_json(base_url, f"/api/generate/{job_id}")
        if status.get("status") in {"done", "error"}:
            return status
        time.sleep(3)
    raise TimeoutError(f"Generation timed out: {job_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live native vs LanPaint inpaint visual verification.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8200")
    parser.add_argument("--out", default=str(ROOT / "new_inpaint_tests" / "lanpaint_verification"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    out_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    source, mask = _source_and_mask()
    source.save(out_dir / "source.png")
    mask.save(out_dir / "mask.png")
    source_b64 = _b64_png(source)
    mask_b64 = _b64_png(mask)

    common = {
        "prompt": "replace the red object with a glossy blue ceramic vase, studio lighting, realistic shadows, coherent tabletop",
        "negative_prompt": "blurry, smeared, low quality, text, watermark, broken edges",
        "mode": "inpaint",
        "model_profile": "krea_turbo",
        "checkpoint": "turbo",
        "quantization": "fp8",
        "width": 512,
        "height": 512,
        "num_images": 1,
        "seed": 424242,
        "denoise": 1.0,
        "init_image_b64": source_b64,
        "mask_b64": mask_b64,
        "sampler": "euler",
        "scheduler": "simple",
        "cfg": 1.0,
    }

    cases = [
        ("native", {**common, "steps": 8, "inpaint_method": "native"}),
        (
            "lanpaint",
            {
                **common,
                "steps": 20,
                "inpaint_method": "lanpaint_experimental",
                "lanpaint_inner_steps": 5,
                "lanpaint_lambda": 16.0,
                "lanpaint_step_size": 0.2,
                "lanpaint_beta": 1.0,
                "lanpaint_friction": 15.0,
                "lanpaint_early_stop": 1,
                "lanpaint_prompt_mode": "Image First",
            },
        ),
    ]

    previews: list[tuple[str, Image.Image]] = [("source", source), ("mask", mask.convert("RGB"))]
    notes: list[str] = []
    for label, payload in cases:
        try:
            status = _run_case(args.base_url, payload, args.timeout)
            (out_dir / f"{label}_job.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            if status.get("status") != "done":
                notes.append(f"- {label}: failed: {status.get('error')}")
                continue
            image_b64 = status["images"][0]
            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            image.save(out_dir / f"{label}.png")
            previews.append((label, image))
            notes.append(f"- {label}: completed.")
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            notes.append(f"- {label}: skipped/failed: {exc}")

    _save_contact_sheet(previews, out_dir / "contact_sheet.png")
    review = [
        "# LanPaint Inpaint Visual Review",
        "",
        "Fixed seed prompt: replace red object with glossy blue ceramic vase.",
        "",
        "## Results",
        *notes,
        "",
        "## Default Recommendation",
        "- Native Krea remains the safest default.",
        "- LanPaint should be marked experimental but usable for hard-mask inpaint with 20+ diffusion steps, 5 think steps, lambda 16, step size 0.2, beta 1, friction 15, early stop 1.",
        "- Use hard binary masks for LanPaint. Soft masks are converted to binary before sampling.",
    ]
    (out_dir / "VISUAL_REVIEW.md").write_text("\n".join(review), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
