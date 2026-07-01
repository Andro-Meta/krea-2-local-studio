from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from inference import Krea2Pipeline  # noqa: E402
from pid_decoder_provider import upscale_pid  # noqa: E402
from quality_assets import asset_by_id  # noqa: E402
from schemas import GenerationRequest  # noqa: E402
from settings import settings  # noqa: E402


SCENES = [
    {
        "slug": "grunge_teen_bedroom",
        "prompt": (
            "a candid 1990s film photo of a messy grunge teen bedroom, Super Nintendo and Sega Genesis "
            "on the floor near a tube TV, two teenagers sitting on the carpet playing Sonic on the Sega Genesis, "
            "band posters, plaid blanket, stacks of game cartridges, warm lamp light, disposable camera flash, "
            "authentic 90s bedroom aesthetic, natural imperfect photo"
        ),
    },
    {
        "slug": "mall_food_court",
        "prompt": (
            "a candid 1990s film photo of three teenagers at a mall food court, paper soda cups and pizza slices "
            "on a plastic tray, colorful neon signs, denim jackets, scrunchies, grainy disposable camera look, "
            "authentic 90s social snapshot with natural expressions"
        ),
    },
    {
        "slug": "garage_band",
        "prompt": (
            "a candid 1990s film photo of a teenage garage band practicing in a cluttered suburban garage, "
            "two people with guitars and one drummer, old amplifier, posters, concrete floor, warm tungsten bulb, "
            "grainy 35mm snapshot, authentic 90s alternative music scene"
        ),
    },
]

SEED = 1994
WIDTH = 512
HEIGHT = 512
STEPS = 4
CFG = 0.0
SAMPLER = "euler"
SCHEDULER = "simple"


def output_dir() -> Path:
    existing = os.environ.get("KREA_COMPARE_DIR", "").strip()
    if existing:
        out = Path(existing)
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "outputs" / f"vae_pid_comparison_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_font(size: int):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_request(prompt: str) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt,
        diffusion_engine="native_int8_convrot",
        checkpoint="turbo",
        quantization="int8",
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg=CFG,
        sampler=SAMPLER,
        scheduler=SCHEDULER,
        seed=SEED,
        num_images=1,
        use_rebalance=False,
        use_prompt_expander=False,
        krea_enhancer_enabled=False,
    )


def b64_to_image(value: str) -> Image.Image:
    import base64
    import io

    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def save_panel(path: Path, scene: dict, panels: list[tuple[str, Image.Image, float]]) -> None:
    tile_w, tile_h = 512, 512
    header_h = 88
    label_h = 54
    font_title = load_font(20)
    font_label = load_font(18)
    font_small = load_font(13)
    canvas = Image.new("RGB", (tile_w * len(panels), header_h + tile_h + label_h), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    title = f"{scene['slug']} | seed {SEED} | {STEPS} steps | {SAMPLER}/{SCHEDULER} | {WIDTH}x{HEIGHT}"
    draw.text((16, 12), title, fill=(245, 245, 245), font=font_title)
    draw.text((16, 42), scene["prompt"][:180], fill=(190, 190, 195), font=font_small)
    for idx, (label, image, seconds) in enumerate(panels):
        x = idx * tile_w
        thumb = image.resize((tile_w, tile_h), Image.LANCZOS)
        canvas.paste(thumb, (x, header_h))
        draw.rectangle((x, header_h + tile_h, x + tile_w, header_h + tile_h + label_h), fill=(35, 35, 42))
        draw.text((x + 12, header_h + tile_h + 8), label, fill=(255, 255, 255), font=font_label)
        draw.text((x + 12, header_h + tile_h + 31), f"{seconds:.2f}s", fill=(205, 205, 210), font=font_small)
    canvas.save(path)


def generate_batch(label: str, vae_path: str, scenes: list[dict], out: Path) -> tuple[dict[str, Image.Image], dict[str, float]]:
    existing_images: dict[str, Image.Image] = {}
    existing_times: dict[str, float] = {}
    for scene in scenes:
        path = out / f"{scene['slug']}_{label}.png"
        if path.exists():
            existing_images[scene["slug"]] = Image.open(path).convert("RGB")
            existing_times[scene["slug"]] = 0.0
    if len(existing_images) == len(scenes):
        print(f"{label} reuse existing outputs", flush=True)
        return existing_images, existing_times

    settings.krea2_vae_path = vae_path
    pipeline = Krea2Pipeline()
    checkpoint = str(asset_by_id("krea2_turbo_int8_convrot").local_path)
    started = time.time()
    pipeline.load(checkpoint, "int8")
    print(f"{label} load {time.time() - started:.2f}s ({getattr(pipeline.ae, 'vae_source', '')})", flush=True)
    images: dict[str, Image.Image] = {}
    times: dict[str, float] = {}
    try:
        for scene in scenes:
            req = make_request(scene["prompt"])
            started = time.time()
            result_b64, _seed, filenames, _lora_reports, metadata = pipeline.generate(req)
            elapsed = time.time() - started
            image = b64_to_image(result_b64[0])
            image.save(out / f"{scene['slug']}_{label}.png")
            images[scene["slug"]] = image
            times[scene["slug"]] = elapsed
            print(f"{label} {scene['slug']} {elapsed:.2f}s {filenames} {metadata[0].get('extra', {}).get('quality')}", flush=True)
    finally:
        pipeline.unload()
        del pipeline
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return images, times


def main() -> None:
    out = output_dir()
    qwen_images, qwen_times = generate_batch("qwen", "", SCENES, out)
    wan_path = str(asset_by_id("wan_2_1_vae").local_path)
    wan_images, wan_times = generate_batch("wan21", wan_path, SCENES, out)
    pid_settings = __import__("main")._pid_settings()
    pid_images: dict[str, Image.Image] = {}
    pid_times: dict[str, float] = {}
    for scene in SCENES:
        started = time.time()
        image = upscale_pid(qwen_images[scene["slug"]], pid_settings, prompt=scene["prompt"], scale=4)
        elapsed = time.time() - started
        image.save(out / f"{scene['slug']}_pid_full.png")
        pid_images[scene["slug"]] = image
        pid_times[scene["slug"]] = elapsed
        print(f"pid {scene['slug']} {elapsed:.2f}s", flush=True)
        save_panel(
            out / f"{scene['slug']}_side_by_side.png",
            scene,
            [
                ("Qwen VAE", qwen_images[scene["slug"]], qwen_times[scene["slug"]]),
                ("Wan 2.1 VAE", wan_images[scene["slug"]], wan_times[scene["slug"]]),
                ("PiD 4x (downsampled)", pid_images[scene["slug"]], pid_times[scene["slug"]]),
            ],
        )
    report = {
        "seed": SEED,
        "settings": {"width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfg": CFG, "sampler": SAMPLER, "scheduler": SCHEDULER},
        "scenes": [
            {
                "slug": scene["slug"],
                "prompt": scene["prompt"],
                "qwen_seconds": qwen_times[scene["slug"]],
                "wan21_seconds": wan_times[scene["slug"]],
                "pid_seconds": pid_times[scene["slug"]],
                "side_by_side": f"{scene['slug']}_side_by_side.png",
            }
            for scene in SCENES
        ],
    }
    (out / "timing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"COMPARISON_DIR {out}", flush=True)


if __name__ == "__main__":
    main()
