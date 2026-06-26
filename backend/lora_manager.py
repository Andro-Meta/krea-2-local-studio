from __future__ import annotations

import logging
from pathlib import Path

from settings import LORAS_DIR

logger = logging.getLogger(__name__)

OFFICIAL_LORAS: dict[str, dict] = {
    "krea2_darkbrush":      {"display_name": "Dark Brush",      "trigger_words": ["monochrome ink wash style"], "strength": 1.0},
    "krea2_dotmatrix":      {"display_name": "Dot Matrix",      "trigger_words": ["monochrome stippling style"], "strength": 1.0},
    "krea2_kidsdrawing":    {"display_name": "Kids Drawing",    "trigger_words": ["naive expressive sketch style"], "strength": 1.0},
    "krea2_neondrip":       {"display_name": "Neon Drip",       "trigger_words": ["textured abstract neon style"], "strength": 1.0},
    "krea2_rainywindow":    {"display_name": "Rainy Window",    "trigger_words": ["rainy window style"], "strength": 1.0},
    "krea2_retroanime":     {"display_name": "Retro Anime",     "trigger_words": ["purple retro anime style"], "strength": 1.0},
    "krea2_softwatercolor": {"display_name": "Soft Watercolor", "trigger_words": ["art deco watercolor style"], "strength": 1.0},
    "krea2_sunsetblur":     {"display_name": "Sunset Blur",     "trigger_words": ["ethereal motion blur style"], "strength": 1.0},
    "krea2_vintagetarot":   {"display_name": "Vintage Tarot",   "trigger_words": ["vintage tarot style"], "strength": 1.0},
}

OFFICIAL_LORA_HF_IDS: dict[str, str] = {
    "krea2_darkbrush":      "Comfy-Org/Krea-2",
    "krea2_dotmatrix":      "Comfy-Org/Krea-2",
    "krea2_kidsdrawing":    "Comfy-Org/Krea-2",
    "krea2_neondrip":       "Comfy-Org/Krea-2",
    "krea2_rainywindow":    "Comfy-Org/Krea-2",
    "krea2_retroanime":     "Comfy-Org/Krea-2",
    "krea2_softwatercolor": "Comfy-Org/Krea-2",
    "krea2_sunsetblur":     "Comfy-Org/Krea-2",
    "krea2_vintagetarot":   "Comfy-Org/Krea-2",
}


def list_loras() -> list[dict]:
    results = []
    for name, info in OFFICIAL_LORAS.items():
        path = LORAS_DIR / f"{name}.safetensors"
        results.append({
            "filename": f"{name}.safetensors",
            "name": name,
            "display_name": info.get("display_name", name.replace("krea2_", "").replace("_", " ").title()),
            "trigger_words": info["trigger_words"],
            "strength": info["strength"],
            "is_official": True,
            "installed": path.exists(),
            "compatible": True,
            "match_info": "official Krea-2 LoRA",
        })
    for f in sorted(LORAS_DIR.glob("*.safetensors")):
        if f.stem not in OFFICIAL_LORAS:
            verdict = inspect_lora(f)
            results.append({
                "filename": f.name,
                "name": f.stem,
                "display_name": f.stem.replace("_", " ").title(),
                "trigger_words": [],
                "strength": 1.0,
                "is_official": False,
                "installed": True,
                "compatible": verdict["compatible"],
                "match_info": verdict["reason"],
            })
    return results


def build_trigger_prompt(prompt: str, loras: list[dict]) -> str:
    words: list[str] = []
    for lora in loras:
        if not lora.get("enabled", True):
            continue
        name = lora.get("name", "")
        if name in OFFICIAL_LORAS:
            words.extend(OFFICIAL_LORAS[name]["trigger_words"])
    if words:
        return f"{', '.join(words)}, {prompt}"
    return prompt


# ---------------------------------------------------------------------------
# LoRA application (diffusers/PEFT format → SingleStreamDiT, fp8-aware)
#
# Krea-2 LoRAs ship in diffusers naming (transformer.transformer_blocks.N.attn.to_q
# .lora_A/.lora_B). The model uses official names (blocks.N.attn.wq ...). We map
# names, then apply each LoRA as an ADDITIVE low-rank path in the layer's forward:
#     out = base_forward(x) + scale * (x @ A^T) @ B^T
# This keeps weights fp8-resident (no 24GB bf16 merge), composes with the fp8
# dequant patch and with multiple LoRAs, and is reversible (clear the adapter list).
# ---------------------------------------------------------------------------

_FIXED_NAME_MAP = {
    "img_in":              "first",
    "final_layer.linear":  "last.linear",
    "txt_in.linear_1":     "txtmlp.1",
    "txt_in.linear_2":     "txtmlp.3",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj":       "tproj.1",
}


def _lora_base_to_module(base: str) -> str:
    """Map a diffusers LoRA base name → SingleStreamDiT module name."""
    s = base
    if s.startswith("transformer."):
        s = s[len("transformer."):]
    s = s.replace("transformer_blocks.", "blocks.")
    s = s.replace("text_fusion.", "txtfusion.")
    s = (s.replace(".attn.to_q", ".attn.wq")
          .replace(".attn.to_k", ".attn.wk")
          .replace(".attn.to_v", ".attn.wv")
          .replace(".attn.to_out.0", ".attn.wo")
          .replace(".attn.to_gate", ".attn.gate")
          .replace(".ff.gate", ".mlp.gate")
          .replace(".ff.up", ".mlp.up")
          .replace(".ff.down", ".mlp.down"))
    return _FIXED_NAME_MAP.get(s, s)


def _ensure_lora_wrapper(module) -> None:
    """Wrap module.forward once so it adds the sum of its LoRA paths.

    Chains on top of whatever forward is already set (e.g. the fp8 dequant
    closure), so the LoRA delta is added in the real/dequantized space.
    """
    import torch.nn.functional as F
    if getattr(module, "_lora_adapters", None) is not None:
        return
    module._lora_adapters = []
    base_forward = module.forward

    def lora_forward(x, _base=base_forward, _m=module):
        out = _base(x)
        for A, B, scale in _m._lora_adapters:
            out = out + scale * F.linear(F.linear(x, A), B)
        return out

    module.forward = lora_forward


def clear_loras(model) -> None:
    """Remove all attached LoRA adapters (revert to base weights)."""
    for _, mod in model.named_modules():
        ad = getattr(mod, "_lora_adapters", None)
        if ad is not None:
            ad.clear()


# Supported LoRA key conventions: diffusers/PEFT (lora_A/lora_B) and kohya
# (lora_down/lora_up). Both use the same math (B@A == up@down).
_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight")   # the [rank, in] factor
_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight")     # the [out, rank] factor
_ALPHA_SUFFIX = ".alpha"

# Minimum fraction of a LoRA's layers that must map (with correct shapes) to the
# Krea-2 model for us to be confident it is a real Krea-2 LoRA. A genuine Krea-2
# LoRA matches ~1.0; a wrong-model LoRA (SDXL/Flux/etc.) matches ~0 (names don't
# map, or names collide but shapes differ — e.g. Flux to_q is 3072² vs our 6144²).
_COMPAT_THRESHOLD = 0.5

_MODEL_SHAPES: dict | None = None


def _model_linear_shapes(model=None) -> dict:
    """name -> (out_features, in_features) for every Linear in the DiT.

    Uses the live model when given; otherwise builds the architecture on the
    meta device once (no weights, no VRAM) and caches it for import-time checks.
    """
    global _MODEL_SHAPES
    import torch
    if model is not None:
        return {n: tuple(m.weight.shape) for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)}
    if _MODEL_SHAPES is None:
        import os
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        from krea2.mmdit import SingleStreamDiT, SingleMMDiTConfig
        with torch.device("meta"):
            _m = SingleStreamDiT(SingleMMDiTConfig(
                features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
                multiplier=4, layers=28, patch=2, channels=16, txtlayers=12))
        _MODEL_SHAPES = {n: tuple(mm.weight.shape) for n, mm in _m.named_modules()
                         if isinstance(mm, torch.nn.Linear)}
    return _MODEL_SHAPES


def _split_suffix(key: str, suffixes: tuple) -> str | None:
    for suf in suffixes:
        if key.endswith(suf):
            return key[: -len(suf)]
    return None


def _load_lora_pairs(sd) -> list[tuple]:
    """Group a LoRA state dict into (base, A, B, scale_factor) tuples.

    Handles both diffusers (lora_A/lora_B) and kohya (lora_down/lora_up) naming,
    plus an optional per-module `.alpha` (scale_factor = alpha/rank, else 1.0).
    """
    parts: dict[str, dict] = {}
    for k in sd:
        base = _split_suffix(k, _A_SUFFIXES)
        if base is not None:
            parts.setdefault(base, {})["A"] = sd[k]; continue
        base = _split_suffix(k, _B_SUFFIXES)
        if base is not None:
            parts.setdefault(base, {})["B"] = sd[k]; continue
        if k.endswith(_ALPHA_SUFFIX):
            parts.setdefault(k[: -len(_ALPHA_SUFFIX)], {})["alpha"] = sd[k]
    pairs = []
    for base, d in parts.items():
        if "A" not in d or "B" not in d:
            continue
        A, B = d["A"], d["B"]
        rank = A.shape[0]
        if "alpha" in d:
            a = d["alpha"]
            alpha = float(a.item()) if hasattr(a, "item") else float(a)
        else:
            alpha = float(rank)
        scale_factor = (alpha / rank) if rank else 1.0
        pairs.append((base, A, B, scale_factor))
    return pairs


def inspect_lora(path, model=None) -> dict:
    """Strictly check whether a safetensors file is a real Krea-2 LoRA.

    Reads only tensor headers (no values). Returns format, matched/total layer
    counts, a `compatible` verdict, and a human-readable reason.
    """
    from safetensors import safe_open
    shapes = _model_linear_shapes(model)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            has_ab = any(k.endswith(".lora_A.weight") for k in keys)
            has_updown = any(k.endswith(".lora_down.weight") for k in keys)
            fmt = "diffusers" if has_ab else ("kohya" if has_updown else "unknown")
            a_suf = ".lora_A.weight" if has_ab else ".lora_down.weight"
            b_suf = ".lora_B.weight" if has_ab else ".lora_up.weight"
            keyset = set(keys)
            total = matched = bad_shape = 0
            for k in keys:
                if not k.endswith(a_suf):
                    continue
                base = k[: -len(a_suf)]
                if (base + b_suf) not in keyset:
                    continue
                total += 1
                A = tuple(f.get_slice(k).get_shape())
                B = tuple(f.get_slice(base + b_suf).get_shape())
                sh = shapes.get(_lora_base_to_module(base))
                if sh is None:
                    continue
                out_f, in_f = sh
                if len(A) == 2 and len(B) == 2 and A[1] == in_f and B[0] == out_f and A[0] == B[1]:
                    matched += 1
                else:
                    bad_shape += 1
    except Exception:  # noqa: BLE001
        logger.exception("LoRA inspection failed")
        return {"format": "unknown", "total": 0, "matched": 0,
                "compatible": False, "reason": "unreadable or invalid safetensors file"}

    frac = (matched / total) if total else 0.0
    compatible = total > 0 and frac >= _COMPAT_THRESHOLD
    if total == 0:
        reason = "no LoRA tensors found - not a LoRA file"
    elif compatible:
        reason = f"Krea-2 LoRA ({fmt}): {matched}/{total} layers match"
    else:
        reason = (f"not a Krea-2 LoRA: only {matched}/{total} layers match "
                  f"(wrong model or unsupported naming) - will not be applied")
    return {"format": fmt, "total": total, "matched": matched,
            "bad_shape": bad_shape, "compatible": compatible, "reason": reason}


def apply_loras(model, loras: list[dict], device: str = "cuda") -> list[dict]:
    """Reset then attach compatible LoRAs as additive low-rank paths.

    Incompatible LoRAs (wrong model / unsupported format) are skipped, not
    applied — never corrupt the output. Returns a per-LoRA report list.
    """
    clear_loras(model)
    reports: list[dict] = []
    if not loras:
        return reports
    module_by_name = dict(model.named_modules())
    for lora in loras:
        if not lora.get("enabled", True):
            continue
        filename = lora.get("filename") or (lora.get("name", "") + ".safetensors")
        name = lora.get("name") or filename
        path = LORAS_DIR / filename
        if not path.exists():
            reports.append({"name": name, "applied": False, "reason": "file not found"})
            logger.warning(f"LoRA not found: {path}")
            continue
        verdict = inspect_lora(path, model)
        if not verdict["compatible"]:
            reports.append({"name": name, "applied": False, **verdict})
            logger.warning(f"LoRA {filename} NOT applied: {verdict['reason']}")
            continue
        strength = float(lora.get("strength", 1.0))
        try:
            n = _attach_single_lora(model, module_by_name, path, strength, device)
            reports.append({"name": name, "applied": True, "matched": n,
                            "total": verdict["total"], "format": verdict["format"],
                            "reason": f"applied {n} layers"})
            logger.info(f"Applied LoRA {filename} (strength={strength}, {n} layers, {verdict['format']})")
        except Exception as e:  # noqa: BLE001
            reports.append({"name": name, "applied": False, "reason": str(e)})
            logger.error(f"LoRA apply failed {filename}: {e}")
    return reports


def _attach_single_lora(model, module_by_name, path: Path, strength: float, device: str) -> int:
    import torch
    from safetensors.torch import load_file

    dtype = next(model.parameters()).dtype
    sd = load_file(str(path), device=device)
    attached = 0
    for base, A, B, scale_factor in _load_lora_pairs(sd):
        mod = module_by_name.get(_lora_base_to_module(base))
        if mod is None or not isinstance(mod, torch.nn.Linear):
            continue
        out_f, in_f = mod.weight.shape
        if A.shape[1] != in_f or B.shape[0] != out_f or A.shape[0] != B.shape[1]:
            continue
        A = A.to(device=device, dtype=dtype)             # (rank, in)
        B = B.to(device=device, dtype=dtype)             # (out, rank)
        _ensure_lora_wrapper(mod)
        mod._lora_adapters.append((A, B, strength * scale_factor))
        attached += 1
    if attached == 0:
        raise RuntimeError("no LoRA layers matched the model")
    return attached
