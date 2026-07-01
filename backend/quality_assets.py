from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from settings import LOCAL_AI_DIR, MODELS_DIR


@dataclass(frozen=True)
class QualityAssetSpec:
    id: str
    repo_id: str
    filename: str | None
    local_path: Path
    kind: Literal["file", "snapshot"]
    purpose: str
    allow_patterns: list[str] | None = None
    download_enabled: bool = True
    disabled_reason: str = ""


def asset_specs() -> list[QualityAssetSpec]:
    diffusion_dir = MODELS_DIR / "krea2" / "diffusion_models"
    vae_dir = MODELS_DIR / "krea2" / "vae"
    text_encoder_dir = MODELS_DIR / "krea2" / "text_encoders"
    gguf_dir = MODELS_DIR / "gguf"
    return [
        QualityAssetSpec(
            id="krea2_turbo_bf16",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_turbo_bf16.safetensors",
            local_path=diffusion_dir / "krea2_turbo_bf16.safetensors",
            kind="file",
            purpose="Krea 2 Turbo BF16 benchmark and quality comparison",
        ),
        QualityAssetSpec(
            id="krea2_raw_bf16",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_raw_bf16.safetensors",
            local_path=diffusion_dir / "krea2_raw_bf16.safetensors",
            kind="file",
            purpose="Krea 2 RAW BF16 benchmark and research-quality comparison",
        ),
        QualityAssetSpec(
            id="krea2_raw_fp8",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_raw_fp8_scaled.safetensors",
            local_path=diffusion_dir / "krea2_raw_fp8_scaled.safetensors",
            kind="file",
            purpose="Krea 2 RAW pre-quantized fp8 — loads on 24GB VRAM without RAM-heavy dynamic quant",
        ),
        QualityAssetSpec(
            id="krea2_turbo_int8_convrot",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_turbo_int8_convrot.safetensors",
            local_path=diffusion_dir / "krea2_turbo_int8_convrot.safetensors",
            kind="file",
            purpose="Krea 2 Turbo INT8 ConvRot for native PyTorch W8A8 loading",
        ),
        QualityAssetSpec(
            id="krea2_turbo_int8",
            repo_id="Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8",
            filename="krea2_turbo_int8.safetensors",
            local_path=diffusion_dir / "krea2_turbo_int8.safetensors",
            kind="file",
            purpose="Krea 2 Turbo regular INT8 for current ComfyUI native INT8 benchmarking",
        ),
        QualityAssetSpec(
            id="krea2_raw_int8_convrot",
            repo_id="Comfy-Org/Krea-2",
            filename="diffusion_models/krea2_raw_int8_convrot.safetensors",
            local_path=diffusion_dir / "krea2_raw_int8_convrot.safetensors",
            kind="file",
            purpose="Krea 2 RAW INT8 ConvRot for native PyTorch W8A8 loading",
        ),
        QualityAssetSpec(
            id="krea2_raw_int8",
            repo_id="Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8",
            filename="krea2_base_int8.safetensors",
            local_path=diffusion_dir / "krea2_base_int8.safetensors",
            kind="file",
            purpose="Krea 2 RAW/Base regular INT8 for current ComfyUI native INT8 benchmarking",
        ),
        QualityAssetSpec(
            id="qwen_image_hdr_vae",
            repo_id="Kijai/QwenImage_experimental",
            filename="qwen_image_HDR_vae_fp32_comfy.safetensors",
            local_path=MODELS_DIR / "krea2" / "vae" / "qwen_image_HDR_vae_fp32_comfy.safetensors",
            kind="file",
            purpose="Optional sharper/HDR Qwen Image VAE for decode (experimental override)",
        ),
        QualityAssetSpec(
            id="wan_2_1_vae",
            repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            filename="split_files/vae/wan_2.1_vae.safetensors",
            local_path=vae_dir / "wan_2.1_vae.safetensors",
            kind="file",
            purpose="Wan 2.1 VAE used by stable-diffusion.cpp Krea2 and some Comfy workflows",
        ),
        QualityAssetSpec(
            id="qwen3vl_abliterated_fp8",
            repo_id="ahmed22xa/Huihui-Qwen3-VL-4B-Instruct-abliterated-comfy",
            filename="Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors",
            local_path=text_encoder_dir / "Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors",
            kind="file",
            purpose="Abliterated Qwen3-VL FP8 text encoder from the referenced Krea2 Turbo workflow (experimental, higher safety risk)",
        ),
        QualityAssetSpec(
            id="pid_gemma_text_encoder",
            repo_id="Comfy-Org/PixelDiT",
            filename="text_encoders/gemma_2_2b_it_elm_bf16.safetensors",
            local_path=text_encoder_dir / "gemma_2_2b_it_elm_bf16.safetensors",
            kind="file",
            purpose="PiD/PixelDiT Gemma text encoder used by the optional PiD high-resolution decoder",
        ),
        QualityAssetSpec(
            id="pid_qwenimage_decoder",
            repo_id="Comfy-Org/PixelDiT",
            filename="diffusion_models/pid_qwenimage_1024_to_4096_4step_bf16.safetensors",
            local_path=diffusion_dir / "pid_qwenimage_1024_to_4096_4step_bf16.safetensors",
            kind="file",
            purpose="PiD Qwen-Image 1024-to-4096 4-step pixel diffusion decoder",
        ),
        QualityAssetSpec(
            id="pid_qwenimage_official_checkpoint",
            repo_id="nvidia/PiD",
            filename="checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
            local_path=MODELS_DIR / "pid" / "checkpoints" / "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step" / "model_ema_bf16.pth",
            kind="file",
            purpose="Official PiD Qwen-Image checkpoint required by the native PiD runtime",
        ),
        QualityAssetSpec(
            id="pid_qwenimage_vae_2d",
            repo_id="nvidia/PiD",
            filename="checkpoints/QwenImage_VAE_2d.pth",
            local_path=MODELS_DIR / "pid" / "checkpoints" / "QwenImage_VAE_2d.pth",
            kind="file",
            purpose="Official 2D Qwen-Image VAE tokenizer required by PiD from-clean runtime",
        ),
        QualityAssetSpec(
            id="krea2_realism_v1_lora",
            repo_id="RudySen/Krea2-realism-V1",
            filename="Krea2-realism-V1.safetensors",
            local_path=MODELS_DIR / "loras" / "Krea2-realism-V1.safetensors",
            kind="file",
            purpose="Krea2 realism LoRA used by the workflow at strength 0.6",
        ),
        QualityAssetSpec(
            id="krea2_filter_bypass",
            repo_id="Kutches/Kr3a",
            filename="krea2filterbypass3.safetensors",
            local_path=MODELS_DIR / "loras" / "krea2filterbypass3.safetensors",
            kind="file",
            purpose="Filter-bypass LoRA referenced by the workflow at strength 4.0",
            download_enabled=False,
            disabled_reason="Blocked: this asset is explicitly intended to bypass safety/filter behavior and should not be auto-downloaded by Krea Studio.",
        ),
        QualityAssetSpec(
            id="gguf_krea2_turbo_q4km",
            repo_id="Abiray/Krea-2-Turbo-GGUF",
            filename="Krea-2-Turbo-Q4_K_M.gguf",
            local_path=gguf_dir / "Krea-2-Turbo-Q4_K_M.gguf",
            kind="file",
            purpose="Recommended GGUF Krea2 Turbo baseline for external stable-diffusion.cpp runtime",
        ),
        QualityAssetSpec(
            id="gguf_krea2_turbo_q3km",
            repo_id="Abiray/Krea-2-Turbo-GGUF",
            filename="Krea-2-Turbo-Q3_K_M.gguf",
            local_path=gguf_dir / "Krea-2-Turbo-Q3_K_M.gguf",
            kind="file",
            purpose="Smallest practical Krea2 Turbo GGUF candidate for low-VRAM/realtime experiments",
        ),
        QualityAssetSpec(
            id="gguf_qwen3vl_4b_q4km",
            repo_id="Qwen/Qwen3-VL-4B-Instruct-GGUF",
            filename="Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            local_path=gguf_dir / "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            kind="file",
            purpose="Qwen3-VL 4B GGUF LLM text encoder for stable-diffusion.cpp Krea2 runtime",
        ),
        QualityAssetSpec(
            id="flux_fill",
            repo_id="black-forest-labs/FLUX.1-Fill-dev",
            filename=None,
            local_path=LOCAL_AI_DIR / "flux1_fill_dev",
            kind="snapshot",
            purpose="FLUX Fill strict inpaint/outpaint provider",
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "tokenizer_2/*",
                "text_encoder/*",
                "text_encoder_2/*",
                "transformer/*",
                "vae/*",
                "*.safetensors",
                "*.json",
                "*.txt",
            ],
        ),
    ]


def asset_by_id(asset_id: str) -> QualityAssetSpec:
    for spec in asset_specs():
        if spec.id == asset_id:
            return spec
    raise KeyError(asset_id)


def asset_installed(spec: QualityAssetSpec) -> bool:
    if spec.kind == "file":
        return spec.local_path.exists()
    return (spec.local_path / "model_index.json").exists()


def asset_status(spec: QualityAssetSpec, *, has_hf_token: bool = False) -> dict:
    installed = asset_installed(spec)
    needs_token = spec.id == "flux_fill" and not installed and not has_hf_token
    return {
        "id": spec.id,
        "repo_id": spec.repo_id,
        "filename": spec.filename,
        "local_path": str(spec.local_path),
        "purpose": spec.purpose,
        "installed": installed,
        "needs_token": needs_token,
        "gated": spec.id == "flux_fill",
        "setup_url": f"https://huggingface.co/{spec.repo_id}",
        "download_enabled": bool(spec.download_enabled),
        "disabled_reason": spec.disabled_reason,
    }


def download_asset(spec: QualityAssetSpec, *, token: str | None = None) -> Path:
    import os

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    spec.local_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.kind == "file":
        from huggingface_hub import hf_hub_download

        if spec.filename is None:
            raise ValueError(f"{spec.id} has no filename")
        downloaded = Path(
            hf_hub_download(repo_id=spec.repo_id, filename=spec.filename, token=token or None)
        )
        if downloaded.resolve() != spec.local_path.resolve():
            shutil.copy2(downloaded, spec.local_path)
        return spec.local_path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(spec.local_path),
            token=token or None,
            allow_patterns=spec.allow_patterns,
            local_dir_use_symlinks=False,
        )
    )
