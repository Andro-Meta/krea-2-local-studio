"""ComfyUI-aware sampler / scheduler catalog.

When the ComfyUI backend is active, the frontend should offer the *full* set of
samplers and schedulers ComfyUI actually exposes on the KSampler node -- which,
with RES4LYF installed, includes res_2s/res_2m/seeds_2, bong_tangent, beta57 and
friends. This module fetches the live KSampler enums from ComfyUI (cached),
decorates the well-known ones with labels/notes, and ships an expanded set of
community-tested combos harvested from the Banodoco Krea-2 Discord.

The shape matches sampler_registry.sampler_catalog so the existing
frontend (ParameterSection) consumes it unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from comfy_client import object_info

logger = logging.getLogger("krea2.comfy")

_TTL = 60.0
_CACHE: dict[str, Any] = {"t": 0.0, "samplers": [], "schedulers": []}

# Fallback enums (captured from ComfyUI 0.27 + RES4LYF) if /object_info is down.
_FALLBACK_SAMPLERS = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "dpm_2", "dpm_2_ancestral", "lms", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m",
    "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddpm", "lcm", "ipndm", "ipndm_v",
    "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
    "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
    "res_2s", "res_3s", "res_2m", "res_3m", "exp_heun_2_x0_sde",
]
_FALLBACK_SCHEDULERS = [
    "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta",
    "normal", "linear_quadratic", "kl_optimal", "bong_tangent", "beta57",
]

# Preferred display order (recommended first). Anything not listed is appended.
_SAMPLER_ORDER = [
    "euler_flow", "euler", "er_sde", "euler_ancestral", "euler_ancestral_cfg_pp", "euler_cfg_pp",
    "res_2s", "res_2m", "res_3s", "res_3m", "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "seeds_2", "seeds_3", "gradient_estimation",
    "gradient_estimation_cfg_pp", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "dpmpp_2s_ancestral",
    "dpmpp_sde", "deis", "ipndm", "ipndm_v", "uni_pc", "uni_pc_bh2", "sa_solver", "sa_solver_pece",
    "ddim", "ddpm", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "lcm", "exp_heun_2_x0_sde",
]

# id -> (label, note). Unlisted samplers get a prettified label.
_SAMPLER_META: dict[str, tuple[str, str]] = {
    "euler_flow": ("Euler (Krea default)", "The shipped Krea Turbo recipe. euler/simple is the frozen turbo schedule."),
    "euler": ("Euler", "Deterministic flow Euler. Robust general-purpose default."),
    "er_sde": ("ER-SDE (community #1)", "The Krea community's most-used sampler on Discord. er_sde/simple @8 or er_sde/beta @10."),
    "euler_ancestral": ("Euler Ancestral", "Re-injects noise each step for extra variation/detail. Great on RAW."),
    "euler_ancestral_cfg_pp": ("Euler Ancestral CFG++", "CFG++ ancestral. Lets you drop CFG ~1-2 pts; less over-saturation. Community favorite for adherence."),
    "euler_cfg_pp": ("Euler CFG++", "Deterministic CFG++. Cleaner high-CFG renders on RAW."),
    "res_2s": ("RES 2S (RES4LYF, 2nd-order) \u2697\ufe0f", "Exponential-RK, 2 model calls/step. Pairs with bong_tangent. The cited 'max quality' sampler."),
    "res_2m": ("RES 2M (RES4LYF multistep) \u2697\ufe0f", "Multistep RES; cheaper than res_2s. Try with beta57."),
    "res_3s": ("RES 3S (RES4LYF, 3rd-order) \u2697\ufe0f", "3rd-order exponential-RK. Slow but very smooth."),
    "res_3m": ("RES 3M (RES4LYF multistep) \u2697\ufe0f", "3rd-order multistep RES4LYF."),
    "res_multistep": ("RES Multistep", "Core ComfyUI RES multistep."),
    "res_multistep_cfg_pp": ("RES Multistep CFG++", "RES multistep with CFG++ guidance."),
    "res_multistep_ancestral": ("RES Multistep Ancestral", "Ancestral RES multistep for extra variation."),
    "seeds_2": ("SEEDS 2 (RES4LYF SDE) \u2697\ufe0f", "Stochastic exponential SDE solver, 2nd-order."),
    "seeds_3": ("SEEDS 3 (RES4LYF SDE) \u2697\ufe0f", "Stochastic exponential SDE solver, 3rd-order."),
    "gradient_estimation": ("Gradient Estimation \u2697\ufe0f", "Gradient-estimation solver; crisp on few steps."),
    "gradient_estimation_cfg_pp": ("Gradient Estimation CFG++ \u2697\ufe0f", "Gradient estimation with CFG++."),
    "dpmpp_2m": ("DPM++ 2M", "Multistep DPM++; strong all-rounder on RAW."),
    "dpmpp_2m_sde": ("DPM++ 2M SDE", "Stochastic DPM++ 2M; add karras/beta for detail."),
    "dpmpp_3m_sde": ("DPM++ 3M SDE", "3rd-order stochastic DPM++."),
    "dpmpp_2s_ancestral": ("DPM++ 2S Ancestral", "Ancestral single-step DPM++."),
    "dpmpp_sde": ("DPM++ SDE", "Single-step stochastic DPM++."),
    "deis": ("DEIS", "Diffusion exponential integrator."),
    "ipndm": ("iPNDM", "Improved pseudo-linear multistep."),
    "ipndm_v": ("iPNDM_v", "Variable-step iPNDM."),
    "uni_pc": ("UniPC", "Unified predictor-corrector; efficient on RAW."),
    "uni_pc_bh2": ("UniPC BH2", "UniPC with BH2 corrector."),
    "sa_solver": ("SA-Solver", "Stochastic Adams solver."),
    "sa_solver_pece": ("SA-Solver PECE", "SA-Solver predict-eval-correct-eval."),
    "ddim": ("DDIM", "Deterministic DDIM. Community uses ddim/beta57 @8 on Turbo."),
    "ddpm": ("DDPM", "Ancestral DDPM."),
    "heun": ("Heun", "2nd-order Heun."),
    "heunpp2": ("Heun++ 2", "Improved Heun."),
    "dpm_2": ("DPM 2", "2nd-order DPM."),
    "dpm_2_ancestral": ("DPM 2 Ancestral", "Ancestral 2nd-order DPM."),
    "lms": ("LMS", "Linear multistep."),
    "lcm": ("LCM", "Only for LCM-distilled models/LoRAs."),
    "exp_heun_2_x0_sde": ("Experimental Heun x0 SDE \u2697\ufe0f", "2nd-order Heun x0 SDE from Comfy detail-refine workflows."),
}

# id -> (label, note, recommended)
_SCHED_META: dict[str, tuple[str, str, bool]] = {
    "simple": ("Simple (Krea flow default)", "Uniform in flow-time after the mu time-shift. Safe baseline.", True),
    "normal": ("Normal", "Uniform-in-time; identical to Simple for Krea flow.", True),
    "beta": ("Beta (crisper detail)", "Beta(0.6,0.6) U-shaped spacing. Most-cited 'sharper' scheduler for flow.", True),
    "beta57": ("Beta57 (RES4LYF Xperiment)", "Beta(0.5,0.7) from the Krea2 Turbo Xperiment workflow. Great with er_sde/ddim.", True),
    "bong_tangent": ("Bong Tangent (RES4LYF)", "Tangent S-curve; clusters steps mid-trajectory. Community pairs with res_2s / euler_ancestral_cfg_pp.", True),
    "sgm_uniform": ("SGM Uniform", "Uniform but drops sigma_min endpoint; denser near the clean end.", True),
    "ddim_uniform": ("DDIM Uniform", "Uniform DDIM spacing. Use with ddim.", True),
    "kl_optimal": ("KL Optimal", "KL-optimal spacing; smooth results.", True),
    "linear_quadratic": ("Linear Quadratic", "Linear-then-quadratic spacing.", False),
    "karras": ("Karras (EDM, experimental for flow)", "EDM-shaped (rho=7). Designed for EPS/EDM; experimental on flow.", False),
    "exponential": ("Exponential (EDM, experimental for flow)", "Log-linear EDM spacing. Experimental on flow.", False),
}


def _pretty(sid: str) -> str:
    return sid.replace("_", " ").title()


def _is_turbo(profile: str) -> bool:
    p = str(profile or "").lower()
    return "turbo" in p or p in ("", "krea_turbo")


def _fetch() -> tuple[list[str], list[str]]:
    now = time.time()
    if _CACHE["samplers"] and (now - _CACHE["t"]) < _TTL:
        return _CACHE["samplers"], _CACHE["schedulers"]
    samplers, schedulers = list(_FALLBACK_SAMPLERS), list(_FALLBACK_SCHEDULERS)
    try:
        req = object_info("KSampler")["KSampler"]["input"]["required"]
        samplers = list(req["sampler_name"][0])
        schedulers = list(req["scheduler"][0])
    except Exception:
        logger.debug("comfy_catalog: falling back to static enums", exc_info=True)
    _CACHE.update(t=now, samplers=samplers, schedulers=schedulers)
    return samplers, schedulers


def _recommended_steps(sampler: str, profile: str) -> int:
    turbo = _is_turbo(profile)
    if sampler == "lcm":
        return 4
    if turbo:
        if sampler in ("er_sde",):
            return 8
        if sampler in ("res_2s", "res_3s", "seeds_2", "seeds_3"):
            return 10
        if sampler in ("euler_ancestral", "euler_ancestral_cfg_pp", "euler_cfg_pp"):
            return 10
        return 8
    base = 28
    if sampler in ("euler_ancestral", "euler_ancestral_cfg_pp", "er_sde", "seeds_2", "seeds_3"):
        base = 30
    if sampler in ("res_2s", "res_3s"):
        base = 24
    return base


def _ordered(samplers: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    # euler_flow is a native alias we always surface first (maps to euler).
    for sid in _SAMPLER_ORDER:
        if sid == "euler_flow" or sid in samplers:
            if sid not in seen:
                out.append(sid)
                seen.add(sid)
    for sid in samplers:
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


# --- Community combos harvested from the Banodoco Krea-2 Discord --------------
# (sampler, scheduler, steps, cfg, label, profile, note, cfg_zero_star)
# NB: Turbo is a distilled, guidance-disabled model -> CFG 1.0 (Discord + HF).
# ComfyUI's cfg=0 denoises toward the unconditional and ignores the prompt, so
# no preset uses CFG 0. Real guidance (CFG>1) is RAW-only; CFG-Zero* presets set
# the cfg_zero_star flag which the UI applies alongside sampler/scheduler.
_COMBOS: tuple[tuple, ...] = (
    # --- Turbo (distilled, ~8 steps, CFG 1, mu frozen to 1024^2) ---
    ("euler_flow", "simple", 8, 1.0, "Turbo default", "turbo", "Shipped Krea Turbo recipe. euler/simple, frozen shift.", False),
    ("er_sde", "simple", 8, 1.0, "Turbo ER-SDE (community #1)", "turbo", "The most-used Discord recipe: er_sde/simple @8.", False),
    ("euler_flow", "beta", 10, 1.0, "Turbo crisp (beta)", "turbo", "Beta spacing squeezes extra detail out of Turbo.", False),
    ("er_sde", "beta57", 6, 1.0, "Xperiment fast (beta57)", "turbo", "Fast Xperiment recipe: er_sde/beta57 @6, CFG 1.", False),
    ("ddim", "beta57", 8, 1.0, "Turbo DDIM \u00d7 beta57", "turbo", "silveroxides combo: ddim/beta57 @8, works cleanly with LoRAs.", False),
    ("euler_ancestral_cfg_pp", "beta", 10, 1.0, "Turbo ancestral CFG++", "turbo", "euler_ancestral_cfg_pp/beta - organic detail at CFG 1.", False),
    ("euler_ancestral", "beta", 10, 1.0, "Turbo ancestral", "turbo", "Ancestral noise adds variation each step. Discord-noted for Turbo.", False),
    ("er_sde", "kl_optimal", 8, 1.0, "Turbo KL-optimal", "turbo", "kl_optimal spacing - Discord-flagged as 'interesting' on Turbo.", False),
    ("dpmpp_sde", "beta", 10, 1.0, "Turbo DPM++ SDE", "turbo", "dpmpp_sde/beta - stochastic detail on Turbo.", False),
    ("res_2s", "bong_tangent", 8, 1.0, "Turbo RES2S \u2697\ufe0f", "turbo", "RES4LYF res_2s + bong_tangent. 2 calls/step; very smooth.", False),
    ("res_2m", "beta57", 8, 1.0, "Turbo RES2M \u2697\ufe0f", "turbo", "RES4LYF res_2m + beta57. Cheaper than res_2s.", False),
    ("res_3s", "bong_tangent", 10, 1.0, "Turbo RES3S \u2697\ufe0f", "turbo", "3rd-order RES4LYF res_3s + bong_tangent.", False),
    ("seeds_2", "simple", 8, 1.0, "Turbo SEEDS2 \u2697\ufe0f", "turbo", "RES4LYF SEEDS2 SDE solver. Crunchier, more variation.", False),
    ("gradient_estimation", "beta", 8, 1.0, "Turbo Gradient Est. \u2697\ufe0f", "turbo", "Gradient-estimation solver; crisp on few steps.", False),
    # --- RAW (base model, CFG 3.5-4, many steps, adaptive shift) ---
    ("euler_flow", "simple", 52, 3.5, "RAW reference (52/3.5)", "raw", "The cited Reddit RAW settings: euler/simple, 52 steps, CFG 3.5.", False),
    ("euler_flow", "beta", 28, 4.0, "RAW crisp", "raw", "Most-cited quality combo - sharper textures than simple.", False),
    ("er_sde", "beta", 10, 4.0, "RAW ER-SDE", "raw", "Community favorite on base: er_sde/beta ~10 steps.", False),
    ("euler_flow", "kl_optimal", 28, 4.0, "RAW KL-optimal", "raw", "kl_optimal spacing on RAW - smooth tonal transitions.", False),
    ("dpmpp_sde", "beta", 28, 4.0, "RAW DPM++ SDE", "raw", "dpmpp_sde/beta on RAW for stochastic detail.", False),
    ("res_2s", "bong_tangent", 24, 4.0, "RAW RES2S max \u2697\ufe0f", "raw", "res_2s + bong_tangent - the cited 'max quality' RAW combo.", False),
    ("res_3m", "beta57", 30, 3.5, "RAW RES3M \u2697\ufe0f", "raw", "3rd-order RES4LYF multistep + beta57.", False),
    ("euler_ancestral_cfg_pp", "beta", 30, 3.0, "RAW CFG++ adherence", "raw", "CFG++ ancestral + beta: best adherence at lower CFG.", False),
    ("euler_cfg_pp", "beta", 28, 5.0, "RAW high-CFG clean", "raw", "Deterministic CFG++ keeps high guidance from over-saturating.", False),
    ("dpmpp_2m_sde", "karras", 28, 4.0, "RAW DPM++ 2M SDE", "raw", "Classic DPM++ 2M SDE / karras workhorse.", False),
    ("uni_pc", "beta", 26, 4.0, "RAW UniPC", "raw", "UniPC predictor-corrector; efficient and clean.", False),
    ("euler_flow", "simple", 8, 2.0, "RAW + Turbo LoRA (fast)", "raw", "Add the Krea Turbo LoRA (~0.6) then use 8 steps / CFG 2 for turbo-speed on RAW.", False),
    # --- CFG-Zero* (arXiv:2503.18886): optimized-scale + zero-init, RAW only ---
    ("euler_ancestral_cfg_pp", "beta", 30, 3.5, "RAW CFG-Zero* adherence \u2728", "raw", "CFG-Zero* corrects velocity error and zero-inits step 1 for cleaner guidance/adherence.", True),
    ("euler_flow", "beta", 28, 4.0, "RAW CFG-Zero* crisp \u2728", "raw", "CFG-Zero* on the crisp beta combo - reduces over-saturation at CFG 4.", True),
)


def _combos(profile: str) -> list[dict]:
    wanted = "turbo" if _is_turbo(profile) else "raw"
    out = []
    for sampler, scheduler, steps, cfg, label, prof, note, cfg_zero in _COMBOS:
        if prof not in (wanted, "any"):
            continue
        entry = {"sampler": sampler, "scheduler": scheduler, "steps": steps,
                 "cfg": cfg, "label": label, "note": note}
        if cfg_zero:
            entry["cfg_zero_star"] = True
        out.append(entry)
    return out


def sampler_catalog(profile: str = "krea_turbo") -> dict:
    samplers, schedulers = _fetch()
    sched_ids = list(schedulers)
    sampler_opts = []
    for sid in _ordered(samplers):
        label, note = _SAMPLER_META.get(sid, (_pretty(sid), ""))
        sampler_opts.append({
            "id": sid,
            "label": label,
            "scheduler": "simple",
            "default_steps": _recommended_steps(sid, profile),
            "default_cfg": 1.0 if _is_turbo(profile) else 4.0,
            "default_denoise": 1.0,
            "supported_schedulers": sched_ids,
            "recommended_steps": _recommended_steps(sid, profile),
            "disabled": False,
            "note": note,
        })
    scheduler_opts = []
    for sid in sched_ids:
        label, note, rec = _SCHED_META.get(sid, (_pretty(sid), "", False))
        scheduler_opts.append({"id": sid, "label": label, "recommended": rec, "note": note})
    return {
        "profile": str(profile or ""),
        "samplers": sampler_opts,
        "schedulers": scheduler_opts,
        "recommended_combos": _combos(profile),
    }
