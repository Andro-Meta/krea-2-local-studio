from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from inference import Krea2Pipeline  # noqa: E402
from quality_assets import asset_by_id  # noqa: E402
from schemas import GenerationRequest  # noqa: E402

PROMPTS = [
    "a candid 1990s film photo of three teenagers at a mall food court, paper soda cups and pizza slices, neon signs, denim jackets, natural expressions",
    "a cinematic portrait photo of an elderly jazz musician backstage, warm tungsten light, textured skin, black suit, brass instrument case",
    "a wide documentary photo of a cluttered artist studio, canvases, paint tubes, morning window light, realistic details, natural color",
]


def first_existing(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for asset_id, quant in candidates:
        path = asset_by_id(asset_id).local_path
        if path.exists():
            return str(path), quant
    raise RuntimeError(f"None of these assets are installed: {[asset for asset, _ in candidates]}")


def first_low_ram_raw_checkpoint() -> tuple[str, str]:
    try:
        return first_existing([
            ("krea2_raw_int8_convrot", "int8"),
            ("krea2_raw_fp8", "fp8"),
        ])
    except RuntimeError as exc:
        raise RuntimeError(
            "RAW + Turbo LoRA benchmark requires a low-RAM RAW checkpoint. "
            "Install krea2_raw_int8_convrot or krea2_raw_fp8 first; RAW BF16 is intentionally not used on this machine."
        ) from exc


def decode_image(value: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def make_request(prompt: str, *, checkpoint: str, quantization: str, loras: list[dict], seed: int, size: int) -> GenerationRequest:
    raw = checkpoint == "raw"
    return GenerationRequest(
        prompt=prompt,
        diffusion_engine="native_int8_convrot" if quantization == "int8" else "native_pytorch",
        checkpoint=checkpoint,
        quantization=quantization,
        width=size,
        height=size,
        steps=52 if raw else 8,
        cfg=3.5 if raw else 0.0,
        mu=None if raw else 1.15,
        sampler="euler",
        scheduler="simple",
        seed=seed,
        num_images=1,
        loras=loras,
        use_rebalance=False,
        use_prompt_expander=False,
        krea_enhancer_enabled=False,
    )


def run_case(label: str, checkpoint_path: str, quantization: str, checkpoint: str, loras: list[dict], prompts: list[str], out: Path, seed: int, size: int) -> dict:
    pipeline = Krea2Pipeline()
    started = time.time()
    pipeline.load(checkpoint_path, quantization)
    load_seconds = time.time() - started
    images: dict[str, str] = {}
    timings: dict[str, float] = {}
    try:
        for index, prompt in enumerate(prompts):
            started = time.time()
            req = make_request(prompt, checkpoint=checkpoint, quantization=quantization, loras=loras, seed=seed + index, size=size)
            result_b64, _seed, filenames, reports, metadata = pipeline.generate(req)
            elapsed = time.time() - started
            image = decode_image(result_b64[0])
            filename = f"{index:02d}_{label}.png"
            image.save(out / filename)
            images[str(index)] = filename
            timings[str(index)] = elapsed
            print(label, index, f"{elapsed:.2f}s", filenames, reports, metadata[0].get("extra", {}).get("quality"), flush=True)
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
    return {
        "label": label,
        "checkpoint_path": checkpoint_path,
        "quantization": quantization,
        "checkpoint": checkpoint,
        "loras": loras,
        "load_seconds": load_seconds,
        "images": images,
        "timings": timings,
    }


def save_panels(out: Path, prompts: list[str], cases: list[dict]) -> None:
    tile = 512
    header = 76
    label_h = 50
    for index, prompt in enumerate(prompts):
        canvas = Image.new("RGB", (tile * len(cases), header + tile + label_h), (20, 20, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f"RAW + Turbo LoRA benchmark | prompt {index}", fill="white")
        draw.text((12, 38), prompt[:180], fill=(190, 190, 195))
        for case_index, case in enumerate(cases):
            x = case_index * tile
            image = Image.open(out / case["images"][str(index)]).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, header))
            draw.rectangle((x, header + tile, x + tile, header + tile + label_h), fill=(35, 35, 42))
            draw.text((x + 10, header + tile + 8), case["label"], fill="white")
            draw.text((x + 10, header + tile + 30), f"{case['timings'][str(index)]:.2f}s", fill=(205, 205, 210))
        canvas.save(out / f"{index:02d}_side_by_side.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Turbo baseline against RAW + K2Q Turbo LoRAs.")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / f"raw_turbo_lora_benchmark_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--single-case", choices=["turbo_baseline", "raw_k2q_r64", "raw_k2q_r128"], default="")
    parser.add_argument("--case-json", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    turbo_path, turbo_quant = first_existing([
        ("krea2_turbo_int8_convrot", "int8"),
        ("krea2_turbo_bf16", "bf16"),
    ])
    raw_path, raw_quant = first_low_ram_raw_checkpoint()

    case_specs = {
        "turbo_baseline": (turbo_path, turbo_quant, "turbo", []),
        "raw_k2q_r64": (raw_path, raw_quant, "raw", [{"name": "k2q_turbo_lora_rank64", "filename": "k2q_turbo_lora_rank64.safetensors", "strength": 0.6, "enabled": True, "block_filter": "all"}]),
        "raw_k2q_r128": (raw_path, raw_quant, "raw", [{"name": "k2q_turbo_lora_rank128", "filename": "k2q_turbo_lora_rank128.safetensors", "strength": 0.6, "enabled": True, "block_filter": "all"}]),
    }
    if args.single_case:
        checkpoint_path, quantization, checkpoint, loras = case_specs[args.single_case]
        case = run_case(args.single_case, checkpoint_path, quantization, checkpoint, loras, PROMPTS, args.out, args.seed, args.size)
        if args.case_json:
            args.case_json.write_text(json.dumps(case, indent=2), encoding="utf-8")
        return

    cases = []
    for label in ["turbo_baseline", "raw_k2q_r64", "raw_k2q_r128"]:
        case_json = args.out / f"{label}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--size", str(args.size),
            "--seed", str(args.seed),
            "--out", str(args.out),
            "--single-case", label,
            "--case-json", str(case_json),
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        cases.append(json.loads(case_json.read_text(encoding="utf-8")))
    save_panels(args.out, PROMPTS, cases)
    report = {"size": args.size, "seed": args.seed, "prompts": PROMPTS, "cases": cases}
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BENCHMARK_DIR {args.out}", flush=True)


if __name__ == "__main__":
    main()
