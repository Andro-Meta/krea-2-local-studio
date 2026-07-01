from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from krea2.sampler_registry import validate_sampler_configuration


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    checkpoint: str
    quantization: str
    text_encoder: str
    vae: str
    default_sampler: str
    scheduler: str = "simple"
    default_steps: int = 8
    default_cfg: float = 1.0
    default_mu: float | None = 1.15
    default_denoise: float = 1.0
    min_denoise: float = 0.0
    max_denoise: float = 1.0
    max_resolution: int = 1536
    min_ram_gb: int = 32
    min_vram_gb: int = 12
    conditioning_mode: str = "auto"
    enabled: bool = True
    disabled_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "krea_turbo": ModelProfile(
        id="krea_turbo",
        label="Krea 2 Turbo",
        checkpoint="turbo",
        quantization="fp8",
        text_encoder="qwen3_vl",
        vae="qwen_image_vae",
        default_sampler="euler",
        default_steps=8,
        default_cfg=1.0,
        default_mu=1.15,
        max_resolution=2048,
        min_ram_gb=16,
        min_vram_gb=10,
    ),
    "krea_raw": ModelProfile(
        id="krea_raw",
        label="Krea 2 RAW",
        checkpoint="raw",
        quantization="bf16",
        text_encoder="qwen3_vl",
        vae="qwen_image_vae",
        default_sampler="euler",
        default_steps=52,
        default_cfg=3.5,
        default_mu=None,
        max_resolution=2048,
        min_ram_gb=48,
        min_vram_gb=24,
    ),
    "qwen_image_edit": ModelProfile(
        id="qwen_image_edit",
        label="Qwen Image Edit",
        checkpoint="custom",
        quantization="bf16",
        text_encoder="qwen_image_edit",
        vae="qwen_image_vae",
        default_sampler="euler",
        default_steps=8,
        default_cfg=1.0,
        conditioning_mode="qwen_image_edit_plus",
        enabled=False,
        disabled_reason="Loader and model layout are not implemented yet.",
    ),
    "lens_turbo": ModelProfile(
        id="lens_turbo",
        label="Lens Turbo",
        checkpoint="custom",
        quantization="bf16",
        text_encoder="gpt_oss_20b",
        vae="flux2_vae",
        default_sampler="euler",
        enabled=False,
        disabled_reason="Requires a distinct GPT-OSS text encoder and Flux2 VAE loader.",
    ),
    "ernie_turbo": ModelProfile(
        id="ernie_turbo",
        label="ERNIE Turbo",
        checkpoint="custom",
        quantization="bf16",
        text_encoder="ernie",
        vae="flux2_vae",
        default_sampler="euler",
        default_mu=3.1,
        enabled=False,
        disabled_reason="Requires ERNIE encoder, Flux2 VAE, and AuraFlow shift support.",
    ),
    "z_image_turbo": ModelProfile(
        id="z_image_turbo",
        label="Z-Image Turbo",
        checkpoint="custom",
        quantization="bf16",
        text_encoder="qwen3_4b",
        vae="ae_safetensors",
        default_sampler="euler",
        enabled=False,
        disabled_reason="Requires a separate Z-Image loader and ae.safetensors VAE path.",
    ),
}


def model_profile_options() -> list[dict[str, Any]]:
    return [profile.to_dict() for profile in MODEL_PROFILES.values()]


def engine_catalog() -> dict[str, Any]:
    native = {
        "engine_id": "native_pytorch",
        "label": "Native PyTorch Krea",
        "default": True,
        "experimental": False,
        "profiles": ["krea_turbo", "krea_raw"],
        "supports_lora": True,
        "supports_style_references": True,
        "supports_moodboards": True,
        "supports_regional_prompts": True,
        "supports_rebalance": True,
        "supports_krea_enhancer": True,
        "supports_flow_samplers": True,
        "supports_standard_samplers": False,
        "supports_cfg": True,
        "supports_img2img": True,
        "supports_inpaint": True,
        "supports_realtime": True,
        "supports_parallel_batch": True,
        "supports_lora_ab_test": False,
        "max_batch": 4,
        "max_resolution": 2048,
        "recommended_steps": 8,
        "unsupported_controls": [],
    }
    native_gguf = {
        **native,
        "engine_id": "native_gguf",
        "label": "Native GGUF",
        "default": False,
        "experimental": False,
        "profiles": ["krea_turbo"],
        "quantization": "gguf",
        "recommended_steps": 8,
        "unsupported_controls": [],
    }
    int8 = {
        **native,
        "engine_id": "native_int8_convrot",
        "label": "Native INT8 ConvRot",
        "default": False,
        "experimental": True,
        "profiles": ["krea_turbo", "krea_raw"],
        "quantization": "int8",
        "supports_lora": True,
        "supports_style_references": True,
        "supports_moodboards": True,
        "supports_regional_prompts": True,
        "supports_rebalance": True,
        "supports_krea_enhancer": True,
        "supports_flow_samplers": True,
        "supports_standard_samplers": False,
        "supports_img2img": True,
        "supports_inpaint": True,
        "supports_realtime": True,
        "supports_parallel_batch": True,
        "supports_lora_ab_test": True,
        "max_batch": 2,
        "max_resolution": 2048,
        "recommended_steps": 8,
        "unsupported_controls": [],
    }
    return {"engines": [native, native_gguf, int8], "default_engine": "native_pytorch"}


def resolve_model_profile(profile_id: str | None, checkpoint: str = "turbo") -> ModelProfile:
    if profile_id:
        profile = MODEL_PROFILES.get(profile_id)
        if profile is None:
            raise ValueError(f"Unknown model profile: {profile_id}")
        return profile
    return MODEL_PROFILES["krea_raw" if str(checkpoint).lower() == "raw" else "krea_turbo"]


def apply_profile_defaults(req) -> ModelProfile:
    profile = resolve_model_profile(getattr(req, "model_profile", ""), getattr(req, "checkpoint", "turbo"))
    if not profile.enabled:
        raise ValueError(f"{profile.label} is not enabled: {profile.disabled_reason}")

    req.model_profile = profile.id
    if not _request_field_was_set(req, "checkpoint"):
        req.checkpoint = profile.checkpoint
    if not _request_field_was_set(req, "quantization"):
        req.quantization = profile.quantization
    if not _request_field_was_set(req, "sampler"):
        req.sampler = profile.default_sampler
    if not _request_field_was_set(req, "scheduler"):
        req.scheduler = profile.scheduler
    if not _request_field_was_set(req, "steps"):
        req.steps = profile.default_steps
    if not _request_field_was_set(req, "cfg"):
        req.cfg = profile.default_cfg
    if not _request_field_was_set(req, "mu"):
        req.mu = profile.default_mu
    if not _request_field_was_set(req, "denoise"):
        req.denoise = profile.default_denoise
    if not _request_field_was_set(req, "conditioning_mode"):
        req.conditioning_mode = profile.conditioning_mode

    validate_sampler_configuration(
        getattr(req, "sampler", profile.default_sampler),
        getattr(req, "scheduler", profile.scheduler),
        profile.id,
    )
    return profile


def _request_field_was_set(req, field: str) -> bool:
    fields = getattr(req, "model_fields_set", None)
    if fields is None:
        fields = getattr(req, "__fields_set__", set())
    return field in fields
