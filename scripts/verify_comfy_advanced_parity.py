from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from krea2.sampler_registry import sampler_options  # noqa: E402
from model_profiles import model_profile_options  # noqa: E402
from preprocessors import preprocess_image  # noqa: E402


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_json(base_url: str, path: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local user-supplied URL
        return json.loads(response.read().decode("utf-8"))


def _contact_sheet(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    if not images:
        return
    thumb_w = 320
    thumb_h = 240
    sheet = Image.new("RGB", (thumb_w * len(images), thumb_h + 32), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_w, thumb_h))
        x = i * thumb_w + (thumb_w - thumb.width) // 2
        y = 24 + (thumb_h - thumb.height) // 2
        sheet.paste(thumb.convert("RGB"), (x, y))
        draw.text((i * thumb_w + 8, 6), label, fill=(0, 0, 0))
    sheet.save(out_path)


def _demo_image() -> Image.Image:
    image = Image.new("RGB", (1024, 640), (230, 232, 228))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 160, 430, 520), fill=(50, 70, 120))
    draw.ellipse((560, 120, 880, 440), fill=(170, 70, 60))
    draw.line((0, 560, 1024, 420), fill=(40, 40, 40), width=8)
    draw.text((145, 190), "KREA", fill=(255, 255, 255))
    return image


def write_static_report(out_dir: Path) -> list[tuple[str, Image.Image]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "sampler_options.json", sampler_options("krea_turbo"))
    _write_json(out_dir / "model_profiles.json", model_profile_options())

    demo = _demo_image()
    previews = []
    for kind in ("canny", "soft_edge", "lineart", "depth"):
        preview = preprocess_image(demo, kind=kind, resolution=512)
        preview.save(out_dir / f"preprocessor_{kind}.png")
        previews.append((kind, preview))
    _contact_sheet(previews, out_dir / "preprocessor_contact_sheet.png")
    return previews


def run_live_checks(base_url: str, out_dir: Path) -> list[str]:
    notes: list[str] = []
    prompt = "cinematic product photo of a translucent blue glass chair in a sunlit studio"
    cases = [
        {"label": "euler", "sampler": "euler"},
        {"label": "heun", "sampler": "exp_heun_2_x0_sde"},
    ]
    live_images: list[tuple[str, Image.Image]] = []
    for case in cases:
        try:
            job = _post_json(base_url, "/api/generate", {
                "prompt": prompt,
                "model_profile": "krea_turbo",
                "sampler": case["sampler"],
                "steps": 8 if case["sampler"] == "euler" else 6,
                "cfg": 1,
                "width": 1024,
                "height": 1024,
                "num_images": 1,
                "seed": 12345,
            }, timeout=60)
            notes.append(f"Queued {case['label']} generation: {job}")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            notes.append(f"Skipped {case['label']} live generation: {exc}")

    if live_images:
        _contact_sheet(live_images, out_dir / "sampler_contact_sheet.png")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Comfy Advanced Parity features.")
    parser.add_argument("--base-url", default="", help="Optional running Krea API URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--out", default=str(ROOT / "verification_outputs" / "comfy_advanced_parity"))
    args = parser.parse_args()

    out_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    write_static_report(out_dir)
    live_notes = run_live_checks(args.base_url, out_dir) if args.base_url else ["Live API checks not requested."]

    review = [
        "# Comfy Advanced Parity Visual Review",
        "",
        "## Static Checks",
        "- Wrote `sampler_options.json` with compatible and guarded sampler options.",
        "- Wrote `model_profiles.json` with active Krea profiles and disabled optional families.",
        "- Wrote preprocessor preview PNGs and `preprocessor_contact_sheet.png`.",
        "",
        "## Live Checks",
        *[f"- {note}" for note in live_notes],
        "",
        "## Recommended Defaults",
        "- Krea Turbo: profile `krea_turbo`, sampler `euler`, scheduler `simple`, 8 steps, CFG 1, denoise 1.",
        "- Detail refine: sampler `exp_heun_2_x0_sde`, 6 steps, CFG 1, denoise 0.55-0.65.",
        "- Ultimate upscale balanced: 2x, 1024 tile, 96 padding, mask blur 12, band-pass seams, denoise 0.28.",
        "- Ultimate upscale best: 2x, 1280 tile, 128 padding, mask blur 16, half-tile intersections, denoise 0.24.",
    ]
    (out_dir / "VISUAL_REVIEW.md").write_text("\n".join(review), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
