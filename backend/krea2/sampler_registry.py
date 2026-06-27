from __future__ import annotations

from dataclasses import dataclass


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


SAMPLER_SPECS: dict[str, SamplerSpec] = {
    "euler": SamplerSpec("euler", "Euler", default_steps=8),
    "euler_flow": SamplerSpec("euler_flow", "Euler Flow (native)", default_steps=8),
    "exp_heun_2_x0_sde": SamplerSpec(
        "exp_heun_2_x0_sde",
        "Experimental Heun x0 SDE",
        default_steps=6,
        default_denoise=0.65,
        note="Flow-matching approximation inspired by Comfy detail-refine workflows.",
    ),
    "lcm": SamplerSpec(
        "lcm",
        "LCM (requires compatible profile)",
        default_steps=4,
        default_denoise=0.3,
        requires_lcm_profile=True,
        note="Only enabled for LCM-compatible model/profile paths.",
    ),
    "dpmpp_2m": SamplerSpec(
        "dpmpp_2m",
        "DPM++ 2M (standard diffusion only)",
        supports_krea_flow=False,
        supports_standard_diffusion=True,
        default_steps=20,
        note="Requires a standard diffusion backend.",
        supported_schedulers=("normal", "karras", "exponential", "simple"),
    ),
    "ddim": SamplerSpec(
        "ddim",
        "DDIM (standard diffusion only)",
        supports_krea_flow=False,
        supports_standard_diffusion=True,
        default_steps=20,
        note="Requires a standard diffusion backend.",
        supported_schedulers=("normal", "karras", "exponential", "simple"),
    ),
    "uni_pc": SamplerSpec(
        "uni_pc",
        "UniPC (standard diffusion only)",
        supports_krea_flow=False,
        supports_standard_diffusion=True,
        default_steps=20,
        note="Requires a standard diffusion backend.",
        supported_schedulers=("normal", "karras", "exponential", "simple"),
    ),
    "lanpaint_experimental": SamplerSpec(
        "lanpaint_experimental",
        "LanPaint experimental",
        default_steps=8,
        note="Inpaint-only masked inner update.",
    ),
}
KREA_FLOW_SAMPLERS = {
    sampler_id for sampler_id, spec in SAMPLER_SPECS.items() if spec.supports_krea_flow and not spec.requires_lcm_profile
}
SAMPLER_ALIASES = {"euler": "euler_flow"}


def normalize_sampler_name(sampler: str) -> str:
    value = str(sampler or "euler_flow").strip()
    return SAMPLER_ALIASES.get(value, value)


def _profile_supports_lcm(profile: str) -> bool:
    return "lcm" in str(profile or "").lower()


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
            "disabled": disabled,
            "note": spec.note,
        })
    return options
