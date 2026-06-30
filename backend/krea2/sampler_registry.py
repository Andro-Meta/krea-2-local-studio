from __future__ import annotations

from dataclasses import dataclass

from .schedulers import ALL_SCHEDULERS

# Native flow samplers accept any flow-time scheduler. simple/normal/beta/
# sgm_uniform are recommended; karras/exponential are EDM-shaped and offered for
# experimentation only.
FLOW_SCHED: tuple[str, ...] = ALL_SCHEDULERS


@dataclass(frozen=True)
class SamplerSpec:
    id: str
    label: str
    scheduler: str = "simple"
    family: str = "krea_flow"
    supports_krea_flow: bool = True
    supports_standard_diffusion: bool = False
    default_steps: int = 8
    default_cfg: float = 1.0
    default_denoise: float = 1.0
    requires_lcm_profile: bool = False
    note: str = ""
    supported_schedulers: tuple[str, ...] = ("simple",)


@dataclass(frozen=True)
class SchedulerSpec:
    id: str
    label: str
    recommended: bool = True
    note: str = ""


SCHEDULER_SPECS: dict[str, SchedulerSpec] = {
    "simple": SchedulerSpec("simple", "Simple (Krea flow default)", note="Uniform in flow-time after the mu time-shift. Safe baseline."),
    "normal": SchedulerSpec("normal", "Normal", note="Uniform-in-time, identical to Simple for Krea flow."),
    "beta": SchedulerSpec("beta", "Beta (crisper detail)", note="Beta(0.6,0.6) U-shaped spacing (arXiv:2407.12173). Most-cited 'sharper' scheduler for flow models."),
    "beta57": SchedulerSpec("beta57", "Beta57 (Xperiment workflow)", note="Beta(0.5,0.7) RES4LYF-style scheduler from the referenced Krea2 Turbo workflow."),
    "sgm_uniform": SchedulerSpec("sgm_uniform", "SGM Uniform", note="Uniform but drops the sigma_min endpoint; slightly denser near the clean end."),
    "bong_tangent": SchedulerSpec("bong_tangent", "Bong Tangent", note="RES4LYF tangent S-curve; clusters steps mid-trajectory. Community pairs it with res_2s / euler_ancestral_cfg_pp."),
    "karras": SchedulerSpec("karras", "Karras (experimental for flow)", recommended=False, note="EDM-shaped (rho=7). Designed for EPS/EDM models; experimental on flow."),
    "exponential": SchedulerSpec("exponential", "Exponential (experimental for flow)", recommended=False, note="Log-linear EDM spacing. Experimental on flow."),
}


SAMPLER_SPECS: dict[str, SamplerSpec] = {
    "euler": SamplerSpec(
        "euler", "Euler (Krea default)", default_steps=8,
        supported_schedulers=FLOW_SCHED,
        note="Deterministic flow Euler. Robust general-purpose default.",
    ),
    "euler_flow": SamplerSpec(
        "euler_flow", "Euler Flow (native)", default_steps=8,
        supported_schedulers=FLOW_SCHED,
        note="Native alias of Euler.",
    ),
    "euler_ancestral": SamplerSpec(
        "euler_ancestral", "Euler Ancestral", default_steps=28,
        supported_schedulers=FLOW_SCHED,
        note="Re-injects fresh noise each step for extra variation/detail. Great on RAW/base.",
    ),
    "euler_ancestral_cfg_pp": SamplerSpec(
        "euler_ancestral_cfg_pp", "Euler Ancestral CFG++", default_steps=28,
        supported_schedulers=FLOW_SCHED,
        note="CFG++ ancestral: step direction from the uncond prediction. Best with real guidance (RAW); lets you lower CFG ~1-2 pts and reduces over-saturation.",
    ),
    "euler_cfg_pp": SamplerSpec(
        "euler_cfg_pp", "Euler CFG++ (deterministic)", default_steps=28,
        supported_schedulers=FLOW_SCHED,
        note="Deterministic CFG++ (no ancestral noise). Cleaner high-CFG renders on RAW.",
    ),
    "er_sde": SamplerSpec(
        "er_sde", "ER-SDE (community favorite)", default_steps=8,
        supported_schedulers=FLOW_SCHED,
        note="Extended Reverse-Time SDE (order-adaptive). The most-used Krea 2 sampler on Discord: er_sde/simple at 8 steps or er_sde/beta at 10.",
    ),
    "res_2s": SamplerSpec(
        "res_2s", "RES 2S (2nd-order)", default_steps=20,
        supported_schedulers=FLOW_SCHED,
        note="2nd-order exponential-RK (RES4LYF res_2s). Two model calls/step. Pairs well with the bong_tangent scheduler.",
    ),
    "exp_heun_2_x0_sde": SamplerSpec(
        "exp_heun_2_x0_sde",
        "Experimental Heun x0 SDE",
        default_steps=6,
        default_denoise=0.65,
        supported_schedulers=FLOW_SCHED,
        note="2nd-order Heun flow approximation inspired by Comfy detail-refine workflows.",
    ),
    "lcm": SamplerSpec(
        "lcm",
        "LCM (requires compatible profile)",
        default_steps=4,
        default_denoise=0.3,
        requires_lcm_profile=True,
        supported_schedulers=("simple", "sgm_uniform"),
        note="Only enabled for LCM-compatible model/profile paths.",
    ),
    "dpmpp_2m": SamplerSpec(
        "dpmpp_2m",
        "DPM++ 2M (standard diffusion only)",
        supports_krea_flow=False,
        supports_standard_diffusion=True,
        default_steps=20,
        note="Requires a standard diffusion backend.",
        supported_schedulers=("normal", "karras", "exponential", "simple", "beta", "sgm_uniform"),
    ),
    "ddim": SamplerSpec(
        "ddim",
        "DDIM (standard diffusion only)",
        supports_krea_flow=False,
        supports_standard_diffusion=True,
        default_steps=20,
        note="Requires a standard diffusion backend.",
        supported_schedulers=("normal", "karras", "exponential", "simple", "ddim_uniform"),
    ),
    "uni_pc": SamplerSpec(
        "uni_pc",
        "UniPC (standard diffusion only)",
        supports_krea_flow=False,
        supports_standard_diffusion=True,
        default_steps=20,
        note="Requires a standard diffusion backend.",
        supported_schedulers=("normal", "karras", "exponential", "simple", "beta"),
    ),
    "lanpaint_experimental": SamplerSpec(
        "lanpaint_experimental",
        "LanPaint experimental",
        default_steps=8,
        supported_schedulers=FLOW_SCHED,
        note="Inpaint-only masked inner update.",
    ),
}
KREA_FLOW_SAMPLERS = {
    sampler_id for sampler_id, spec in SAMPLER_SPECS.items() if spec.supports_krea_flow and not spec.requires_lcm_profile
}
SAMPLER_ALIASES = {"euler": "euler_flow"}


@dataclass(frozen=True)
class ComboSpec:
    sampler: str
    scheduler: str
    steps: int
    cfg: float
    label: str
    profile: str  # "turbo" | "raw" | "any"
    note: str = ""


# Community-favorite sampler/scheduler combos with recommended step counts.
RECOMMENDED_COMBOS: tuple[ComboSpec, ...] = (
    ComboSpec("euler_flow", "simple", 8, 1.0, "Turbo default", "turbo",
              "The shipped Krea Turbo recipe. Fast and reliable."),
    ComboSpec("er_sde", "simple", 8, 1.0, "Turbo ER-SDE", "turbo",
              "The Krea community's most-used recipe — er_sde/simple at 8 steps."),
    ComboSpec("euler_flow", "beta", 10, 1.0, "Turbo crisp", "turbo",
              "Beta spacing squeezes a little extra detail out of Turbo."),
    ComboSpec("er_sde", "beta57", 6, 0.0, "Xperiment fast native", "turbo",
              "Measured native Xperiment default: 6 steps, CFG 0, beta57, Realism LoKr late@0.55. Exact ClownsharKSampler_Beta remains external."),
    ComboSpec("euler_flow", "simple", 28, 4.0, "RAW balanced", "raw",
              "Base/RAW workhorse. Even, predictable."),
    ComboSpec("euler_flow", "beta", 28, 4.0, "RAW crisp detail", "raw",
              "Most-cited Flux quality combo — sharper textures than simple."),
    ComboSpec("er_sde", "beta", 10, 4.0, "RAW ER-SDE", "raw",
              "Community favorite on base: er_sde/beta around 10 steps."),
    ComboSpec("res_2s", "bong_tangent", 24, 4.0, "RAW res_2s max", "raw",
              "2nd-order res_2s + bong_tangent — the cited 'max quality' combo."),
    ComboSpec("euler_ancestral_cfg_pp", "beta", 30, 3.0, "RAW CFG++ adherence", "raw",
              "CFG++ ancestral + beta: best prompt adherence with lower CFG and less burn-in."),
    ComboSpec("euler_ancestral", "beta", 30, 4.0, "RAW variation", "raw",
              "Ancestral noise adds organic variation/detail across seeds."),
    ComboSpec("euler_cfg_pp", "beta", 28, 5.0, "RAW high-CFG clean", "raw",
              "Deterministic CFG++ keeps high guidance from over-saturating."),
)


def normalize_sampler_name(sampler: str) -> str:
    value = str(sampler or "euler_flow").strip()
    return SAMPLER_ALIASES.get(value, value)


def _profile_supports_lcm(profile: str) -> bool:
    return "lcm" in str(profile or "").lower()


def _is_turbo_profile(profile: str) -> bool:
    p = str(profile or "").lower()
    return "turbo" in p or p in ("", "krea_turbo")


def recommended_steps(sampler: str, scheduler: str = "simple", profile: str = "krea_turbo") -> int:
    """Recommended step count for a sampler/scheduler on a given profile.

    Distilled Turbo profiles want few steps; RAW/base want many. Ancestral and
    CFG++ samplers benefit from a couple extra steps on base models.
    """
    name = normalize_sampler_name(sampler)
    if name == "lcm":
        return 4
    if _is_turbo_profile(profile):
        if name == "er_sde":
            return 10 if scheduler == "beta" else 8
        if name == "res_2s":
            return 12
        return 10 if name in ("euler_ancestral", "euler_ancestral_cfg_pp", "euler_cfg_pp") else 8
    base = 28
    if name in ("euler_ancestral", "euler_ancestral_cfg_pp", "er_sde"):
        base = 30
    if name == "res_2s":
        base = 24  # 2 calls/step -> fewer steps for similar cost
    if scheduler == "beta":
        base = max(base, 28)
    return base


def validate_sampler_for_profile(sampler: str, profile: str = "krea_turbo") -> SamplerSpec:
    requested = str(sampler or "euler_flow").strip()
    spec = SAMPLER_SPECS.get(requested) or SAMPLER_SPECS.get(normalize_sampler_name(requested))
    if spec is None:
        raise ValueError(f"Unknown sampler: {sampler}")
    if spec.requires_lcm_profile and not _profile_supports_lcm(profile):
        raise ValueError(f"{spec.label} requires an LCM-compatible model profile.")
    if str(profile or "").startswith("krea") and not spec.supports_krea_flow:
        raise ValueError(f"{spec.label} requires a standard diffusion backend.")
    return spec


def validate_sampler_configuration(
    sampler: str,
    scheduler: str = "simple",
    profile: str = "krea_turbo",
) -> dict[str, str]:
    spec = validate_sampler_for_profile(sampler, profile)
    requested_scheduler = str(scheduler or spec.scheduler).strip()
    if requested_scheduler not in spec.supported_schedulers:
        supported = ", ".join(spec.supported_schedulers)
        raise ValueError(f"{spec.label} does not support scheduler '{requested_scheduler}'. Supported scheduler(s): {supported}.")
    return {
        "sampler": normalize_sampler_name(spec.id),
        "scheduler": requested_scheduler,
        "profile": str(profile or ""),
    }


def sampler_options(profile: str = "krea_turbo") -> list[dict]:
    options = []
    for spec in SAMPLER_SPECS.values():
        disabled = False
        try:
            validate_sampler_for_profile(spec.id, profile)
        except ValueError:
            disabled = True
        options.append({
            "id": spec.id,
            "label": spec.label,
            "scheduler": spec.scheduler,
            "default_steps": spec.default_steps,
            "default_cfg": spec.default_cfg,
            "default_denoise": spec.default_denoise,
            "supported_schedulers": list(spec.supported_schedulers),
            "recommended_steps": recommended_steps(spec.id, spec.scheduler, profile),
            "disabled": disabled,
            "note": spec.note,
        })
    return options


def scheduler_options() -> list[dict]:
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "recommended": spec.recommended,
            "note": spec.note,
        }
        for spec in SCHEDULER_SPECS.values()
    ]


def recommended_combos(profile: str = "krea_turbo") -> list[dict]:
    turbo = _is_turbo_profile(profile)
    wanted = "turbo" if turbo else "raw"
    combos = []
    for c in RECOMMENDED_COMBOS:
        if c.profile not in (wanted, "any"):
            continue
        combos.append({
            "sampler": c.sampler,
            "scheduler": c.scheduler,
            "steps": c.steps,
            "cfg": c.cfg,
            "label": c.label,
            "note": c.note,
        })
    return combos


def sampler_catalog(profile: str = "krea_turbo") -> dict:
    return {
        "profile": str(profile or ""),
        "samplers": sampler_options(profile),
        "schedulers": scheduler_options(),
        "recommended_combos": recommended_combos(profile),
    }
