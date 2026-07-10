"""
Moodboard channel A/B/C/D(/E/F) comparison — how should a moodboard influence a
generation? Same subject + same board + same seeds across the three channels:

  * Baseline : subject prompt only                      (anchor)
  * A_text   : + catalog Qwen TEXT guidance             (current behavior)
  * B_vision : + board IMAGES as Qwen3-VL refs          (images-as-reference)
  * C_both   : text guidance + image refs
  * D_fusion : subject blended with guidance via the magic-prompt expander
  * E_fusion_vision : fused prompt + image refs
  * F_img2prompt    : a board image -> style prompt, blended with subject (+refs)

Everything is generated through ComfyUI nodes (comfy_generate). The only steps
that need the local Qwen VLM (D/E/F prompt prep) are done in ONE upfront phase
with ComfyUI's VRAM explicitly freed first, and the Qwen helper self-unloads
before any generation runs — so two big models never share the 24GB GPU.

Usage:
  python scripts/moodboard_channels_abcd.py --dry-run   # no GPU: resolve board + hydrate images + print plan
  python scripts/moodboard_channels_abcd.py             # full run (needs ComfyUI up)

Outputs land in outputs/moodboard_channels/ (per-condition images + per-seed grid).
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# --------------------------------------------------------------------------- #
# Config — tweak these, then run.
# --------------------------------------------------------------------------- #
SUBJECT = "a portrait of a young woman standing on a city street, looking at the camera"
MOODBOARD_ID = 0          # 0 = auto-pick a board that has Qwen guidance + images
MOODBOARD_UUID = ""       # optional explicit uuid instead of id
N_REF_IMAGES = 3          # board images hydrated as vision refs (engine caps refs at 4)
SEEDS = [111, 222]
W, H = 832, 1152
STEPS = 8
INCLUDE_FUSION = True     # D/E/F need the local Qwen VLM (safely offloaded); set False to skip
EXPANDER_BACKEND = "local"
PANEL_W = 512             # per-panel width in the side-by-side composite


# --------------------------------------------------------------------------- #
# ComfyUI readiness + VRAM helpers
# --------------------------------------------------------------------------- #
def _wait_for_comfy(timeout: float = 180.0) -> bool:
    import comfy_client
    url = comfy_client.comfy_base_url().rstrip("/") + "/system_stats"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


def _vram_free_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9
    except Exception:
        pass
    return -1.0


def _free_comfy_and_wait(target_free_gb: float = 14.0, timeout: float = 30.0) -> None:
    """Evict ComfyUI's resident model so the local Qwen VLM can load without
    fighting it for VRAM (the crash vector). Mirrors the app's pre-helper free."""
    from comfy_client import free_comfy_vram
    from memory_manager import clear_cuda_cache
    print("[mb] freeing ComfyUI VRAM before VLM phase ...", flush=True)
    free_comfy_vram(unload_models=True, free_memory=True)
    clear_cuda_cache()
    t0 = time.time()
    while time.time() - t0 < timeout:
        free = _vram_free_gb()
        if free < 0 or free >= target_free_gb:
            break
        time.sleep(1.5)
    print(f"[mb] VRAM free after evict: {_vram_free_gb():.1f}GB", flush=True)


# --------------------------------------------------------------------------- #
# Phase 0 — resolve the moodboard + hydrate images (network only, no GPU)
# --------------------------------------------------------------------------- #
def _resolve_board() -> dict:
    from moodboards_catalog import get_moodboard, list_moodboards

    async def _pick():
        if MOODBOARD_ID:
            item = await get_moodboard(MOODBOARD_ID)
            return item
        data = await list_moodboards(page_size=300)
        # Prefer a board that has BOTH Qwen text guidance and images so every
        # channel has something to work with.
        best = None
        for it in data.get("items", []):
            guid = (it.get("qwen_guidance") or {}).get("prompt_guidance") or ""
            imgs = it.get("image_urls") or ([it["primary_image_url"]] if it.get("primary_image_url") else [])
            if guid and imgs:
                return it
            if best is None and imgs:
                best = it
        return best

    item = asyncio.run(_pick())
    if not item:
        raise SystemExit("[mb] No moodboard found with images. Add/select one first, or set MOODBOARD_ID.")
    return item


def _hydrate_refs(item: dict, n: int) -> list[str]:
    from moodboards_catalog import _moodboard_image_urls, fetch_moodboard_image_b64
    urls = _moodboard_image_urls([item.get("primary_image_url", ""), *(item.get("image_urls") or [])])
    out: list[str] = []
    for url in urls:
        if len(out) >= n:
            break
        try:
            out.append(fetch_moodboard_image_b64(url))
        except Exception as e:
            print(f"[mb] skip image (fetch failed): {url[:60]} ({e})", flush=True)
    return out


def _board_text(item: dict) -> tuple[str, str, str]:
    """Return (style_text, negative_text, prompt_guidance) for this board."""
    from moodboards_catalog import moodboard_generation_context
    ids = [int(item["id"])] if item.get("id") is not None else []
    uuids = [str(item["uuid"])] if item.get("uuid") else []
    ctx = moodboard_generation_context(ids, moodboard_uuids=uuids)
    guidance = (item.get("qwen_guidance") or {}).get("prompt_guidance") or item.get("taste_profile", "")
    return ctx.get("style_text", ""), ctx.get("negative_text", ""), guidance


# --------------------------------------------------------------------------- #
# Phase 1 — VLM prompt prep (only if fusion conditions are enabled).
# ComfyUI VRAM is freed first; the Qwen helper self-unloads after each call.
# --------------------------------------------------------------------------- #
def _prep_fusion_prompts(item: dict, refs: list[str], guidance: str) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not INCLUDE_FUSION:
        return prompts
    _free_comfy_and_wait()
    from prompt_expander import expand_prompt_result, describe_image_local
    from memory_manager import clear_cuda_cache

    # D/E: blend the subject with the board's guidance, then expand into one prompt.
    seed_text = f"{SUBJECT}. Style direction: {guidance}".strip().rstrip(".") if guidance else SUBJECT
    try:
        fused = expand_prompt_result(seed_text, backend=EXPANDER_BACKEND).expanded.strip()
        prompts["fusion"] = fused or seed_text
        print(f"[mb] fusion prompt ({len(prompts['fusion'])} chars): {prompts['fusion'][:160]}", flush=True)
    except Exception as e:
        print(f"[mb] expander failed, using seed text ({e})", flush=True)
        prompts["fusion"] = seed_text
    clear_cuda_cache()

    # F: read a STYLE paragraph off the first board image, then attach to subject.
    if refs:
        try:
            style = describe_image_local(refs[0], mode="style").get("prompt", "").strip()
            prompts["img2prompt"] = f"{SUBJECT}. {style}".strip() if style else SUBJECT
            print(f"[mb] img2prompt ({len(prompts['img2prompt'])} chars): {prompts['img2prompt'][:160]}", flush=True)
        except Exception as e:
            print(f"[mb] describe failed, skipping F ({e})", flush=True)
    clear_cuda_cache()
    return prompts


# --------------------------------------------------------------------------- #
# Condition builders -> GenerationRequest
# --------------------------------------------------------------------------- #
def _make_req(seed: int, *, prompt: str, ids: list[int], uuids: list[str], images: list[str]):
    from schemas import GenerationRequest
    return GenerationRequest(
        prompt=prompt, steps=STEPS, width=W, height=H, num_images=1, seed=seed,
        moodboard_ids=ids, moodboard_uuids=uuids, moodboard_images=images,
        seed_variance_preset="off",
    )


def _conditions(item: dict, refs: list[str], fusion: dict[str, str]):
    ids = [int(item["id"])] if item.get("id") is not None else []
    uuids = [str(item["uuid"])] if item.get("uuid") else []
    # (key, label, kwargs for _make_req). Setting ids triggers the catalog TEXT
    # append inside comfy_generate; leaving ids empty avoids it.
    conds = [
        ("baseline", "Baseline (subject only)", dict(prompt=SUBJECT, ids=[], uuids=[], images=[])),
        ("A_text", "A: text guidance", dict(prompt=SUBJECT, ids=ids, uuids=uuids, images=[])),
        ("B_vision", "B: image refs", dict(prompt=SUBJECT, ids=[], uuids=[], images=refs)),
        ("C_both", "C: text + image refs", dict(prompt=SUBJECT, ids=ids, uuids=uuids, images=refs)),
    ]
    if INCLUDE_FUSION and fusion.get("fusion"):
        conds.append(("D_fusion", "D: magic-prompt fusion", dict(prompt=fusion["fusion"], ids=[], uuids=[], images=[])))
        conds.append(("E_fusion_vision", "E: fusion + image refs", dict(prompt=fusion["fusion"], ids=[], uuids=[], images=refs)))
    if INCLUDE_FUSION and fusion.get("img2prompt"):
        conds.append(("F_img2prompt", "F: image->prompt + subject", dict(prompt=fusion["img2prompt"], ids=[], uuids=[], images=[])))
    return conds


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def _label_font(size: int):
    from PIL import ImageFont
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _compose(seed: int, panels: list[tuple[str, Path]], out_path: Path) -> None:
    from PIL import Image, ImageDraw
    pw = PANEL_W
    ph = int(pw * H / W)
    bar, title_bar = 54, 46
    font, title_font = _label_font(22), _label_font(28)
    n = len(panels)
    canvas = Image.new("RGB", (pw * n, ph + bar + title_bar), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), f"Moodboard channels  -  seed {seed}", font=title_font, fill=(240, 240, 245))
    for i, (label, img_path) in enumerate(panels):
        x = i * pw
        if img_path.exists():
            img = Image.open(img_path).convert("RGB").resize((pw, ph))
            canvas.paste(img, (x, title_bar))
        draw.rectangle([x, title_bar + ph, x + pw, title_bar + ph + bar], fill=(30, 30, 34))
        tw = draw.textlength(label, font=font)
        draw.text((x + max(6, (pw - tw) / 2), title_bar + ph + 16), label, font=font, fill=(235, 235, 240))
    canvas.save(out_path)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Resolve board + hydrate images + print plan, no GPU work.")
    args = ap.parse_args()

    print("[mb] resolving moodboard ...", flush=True)
    item = _resolve_board()
    style_text, negative_text, guidance = _board_text(item)
    print(f"[mb] board: #{item.get('id')} '{item.get('title')}' uuid={item.get('uuid','')[:8]}", flush=True)
    print(f"[mb]   has guidance: {bool(guidance)} | style_text chars: {len(style_text)} | negatives: {bool(negative_text)}", flush=True)

    print(f"[mb] hydrating up to {N_REF_IMAGES} reference images ...", flush=True)
    refs = _hydrate_refs(item, N_REF_IMAGES)
    print(f"[mb]   hydrated {len(refs)} image(s)", flush=True)

    if args.dry_run:
        print("\n[mb] DRY RUN — planned conditions:")
        for key, label, kw in _conditions(item, refs, {"fusion": "<expander output>", "img2prompt": "<describe output>"} if INCLUDE_FUSION else {}):
            print(f"  - {key:16s} {label:28s} ids={kw['ids']} images={len(kw['images'])} prompt={kw['prompt'][:60]!r}")
        print(f"\n[mb] subject: {SUBJECT}")
        print(f"[mb] guidance: {guidance[:200]}")
        print("[mb] dry run complete — no GPU used.")
        return 0

    if not _wait_for_comfy():
        print("[mb] ComfyUI not reachable. Start it first.", flush=True)
        return 1

    fusion = _prep_fusion_prompts(item, refs, guidance)

    out = ROOT / "outputs" / "moodboard_channels"
    out.mkdir(parents=True, exist_ok=True)
    from comfy_workflows import comfy_generate, OUTPUTS_DIR

    conds = _conditions(item, refs, fusion)
    print(f"[mb] generating {len(conds) * len(SEEDS)} images ...", flush=True)
    saved: dict[int, list[tuple[str, Path]]] = {s: [] for s in SEEDS}
    for key, label, kw in conds:
        for seed in SEEDS:
            req = _make_req(seed, **kw)
            t0 = time.time()
            _r, _seed, filenames, _rep, _meta = comfy_generate(req, save_outputs=True)
            dst = out / f"{key}_seed{seed}.png"
            if filenames:
                src = Path(OUTPUTS_DIR) / filenames[0]
                if src.exists():
                    shutil.copyfile(src, dst)
            saved[seed].append((label, dst))
            print(f"[mb] {key} seed={seed} -> {dst.name} ({time.time() - t0:.1f}s)", flush=True)

    for seed in SEEDS:
        comp = out / f"_compare_seed{seed}.png"
        _compose(seed, saved[seed], comp)
        print(f"[mb] composite -> {comp.name}", flush=True)
    print(f"[mb] done. Folder: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
