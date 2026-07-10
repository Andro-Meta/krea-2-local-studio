"""Generate 4 preview images for each Andro.Meta moodboard, seamless with Krea's
official boards (2:3 portrait, saved locally and served via the custom-images
route). Idempotent: moods that already have 4 previews are skipped unless --force.

Usage (from repo root, with the venv python):
    python scripts/generate_andrometa_previews.py               # all moods
    python scripts/generate_andrometa_previews.py --only retro_web
    python scripts/generate_andrometa_previews.py --limit 3 --force

Requires ComfyUI running (it drives generation via the backend adapter).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from PIL import Image  # noqa: E402
from moods import MOODS  # noqa: E402
from schemas import GenerationRequest  # noqa: E402
from settings import DB_PATH  # noqa: E402
from moodboards_catalog import CUSTOM_MOODBOARD_DIR, _now_iso  # noqa: E402
import comfy_workflows as cw  # noqa: E402

# Krea official moodboard images are 2:3 portrait; 688x1024 is grid-safe (/16).
W, H = 688, 1024
# Four varied subjects so each board reads like a moodboard (consistent style,
# different content) instead of four near-identical frames.
SUBJECTS = [
    "a lone figure in an evocative environment",
    "a sweeping establishing landscape",
    "a close-up still life of characterful objects",
    "an atmospheric architectural interior",
]


def _existing_previews(mood_id: str) -> list[str]:
    d = CUSTOM_MOODBOARD_DIR / f"andrometa-{mood_id}"
    if not d.exists():
        return []
    return sorted(str(p.name) for p in d.glob("img_*.webp"))


def _gen_one(keywords: str, avoids: str, subject: str, seed: int) -> Image.Image:
    req = GenerationRequest(
        prompt=f"{subject}, {keywords}",
        negative_prompt=avoids,
        width=W, height=H, steps=8, cfg=1.0,
        checkpoint="turbo", quantization="fp8", diffusion_engine="native_pytorch",
        seed=seed, mode="txt2img",
    )
    results, *_ = cw.comfy_generate(req, save_outputs=False)
    if not results:
        raise RuntimeError("no image returned")
    raw = base64.b64decode(results[0].split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _update_db(mood_id: str, filenames: list[str]) -> bool:
    uuid = f"andrometa-{mood_id}"
    urls = [f"/api/moodboards/custom-images/{uuid}/{name}" for name in filenames]
    db = sqlite3.connect(str(DB_PATH))
    try:
        cur = db.execute(
            "UPDATE moodboards SET primary_image_url = ?, image_urls = ?, updated_at = ? WHERE uuid = ?",
            (urls[0], json.dumps(urls), _now_iso(), uuid),
        )
        db.commit()
        return cur.rowcount > 0
    finally:
        db.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="single mood id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    moods = [m for m in MOODS if not args.only or m["id"] == args.only]
    if args.limit:
        moods = moods[: args.limit]

    total = len(moods)
    done = skipped = failed = 0
    t0 = time.perf_counter()
    for idx, m in enumerate(moods, 1):
        mid = m["id"]
        out_dir = CUSTOM_MOODBOARD_DIR / f"andrometa-{mid}"
        if not args.force and len(_existing_previews(mid)) >= 4:
            _update_db(mid, _existing_previews(mid))  # ensure DB points at them
            skipped += 1
            print(f"[{idx}/{total}] skip {mid} (already has previews)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        filenames: list[str] = []
        ok = True
        for n, subject in enumerate(SUBJECTS):
            try:
                img = _gen_one(m["keywords"], m["avoids"], subject, seed=1000 + n)
                name = f"img_{n}.webp"
                img.save(out_dir / name, format="WEBP", quality=88, method=4)
                filenames.append(name)
            except Exception as exc:  # keep going; a partial board is still useful
                ok = False
                print(f"    ! {mid} img {n} failed: {exc}")
        if filenames:
            _update_db(mid, filenames)
        if ok and len(filenames) == 4:
            done += 1
        else:
            failed += 1
        rate = (time.perf_counter() - t0) / idx
        print(f"[{idx}/{total}] {mid}: {len(filenames)}/4 images  (~{rate:.0f}s/board, eta {rate*(total-idx)/60:.1f}m)")

    print(f"\nDONE. boards={total} full={done} skipped={skipped} partial/failed={failed} in {(time.perf_counter()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
