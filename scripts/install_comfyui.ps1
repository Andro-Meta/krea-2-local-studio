# Provisions the ComfyUI image engine that the Krea 2 backend drives.
# Idempotent: safe to re-run. Called by install.bat.
#
#   * clones comfy-org/comfyui into <root>/ComfyUI
#   * creates a dedicated venv (prefers Python 3.13, then 3.12)
#   * installs PyTorch (CUDA), ComfyUI requirements, and ComfyUI-Manager
#   * installs the community Krea-2 custom nodes + their requirements
#   * writes extra_model_paths.yaml pointing at <root>/models
#   * writes run_comfyui.bat
param(
    [int]$Port = 8188
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$comfyDir = Join-Path $root "ComfyUI"
$venvDir = Join-Path $comfyDir "venv"
$venvPy = Join-Path $venvDir "Scripts\python.exe"

function Stop-Install($msg) { Write-Host "  ERROR: $msg"; exit 1 }

# -- Preflight: git -----------------------------------------------------------
& git --version *> $null
if ($LASTEXITCODE -ne 0) { Stop-Install "git not found. Install Git for Windows, then re-run install.bat." }

# -- 1. Clone ComfyUI ---------------------------------------------------------
if (-not (Test-Path (Join-Path $comfyDir "main.py"))) {
    Write-Host "  Cloning ComfyUI (comfy-org/comfyui)..."
    & git clone https://github.com/comfy-org/comfyui "$comfyDir"
    if ($LASTEXITCODE -ne 0) { Stop-Install "ComfyUI clone failed." }
} else {
    Write-Host "  ComfyUI already present - skipping clone."
}

# -- 2. venv (prefer Python 3.13, then 3.12) ----------------------------------
if (-not (Test-Path $venvPy)) {
    $pyExe = $null; $pyArgs = @()
    foreach ($v in @("3.13", "3.12")) {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-$v"); break }
    }
    if (-not $pyExe) { $pyExe = "python"; $pyArgs = @() }
    Write-Host "  Creating ComfyUI venv with $pyExe $($pyArgs -join ' ')..."
    & $pyExe @pyArgs -m venv "$venvDir"
    if (-not (Test-Path $venvPy)) { Stop-Install "Failed to create ComfyUI venv." }
}
& $venvPy -m pip install --upgrade pip --quiet

# -- 3. PyTorch (CUDA) --------------------------------------------------------
& $venvPy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyTorch with CUDA (several minutes)..."
    foreach ($cu in @("cu130", "cu128", "cu126")) {
        Write-Host "    trying $cu..."
        & $venvPy -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$cu" --quiet
        & $venvPy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" *> $null
        if ($LASTEXITCODE -eq 0) { break }
    }
    & $venvPy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" *> $null
    if ($LASTEXITCODE -ne 0) { Stop-Install "PyTorch installed but CUDA is unavailable. Update NVIDIA drivers and re-run." }
} else {
    Write-Host "  PyTorch with CUDA already present - skipping."
}

# -- 4. ComfyUI + Manager requirements ----------------------------------------
Write-Host "  Installing ComfyUI requirements..."
& $venvPy -m pip install -r (Join-Path $comfyDir "requirements.txt") --quiet
if ($LASTEXITCODE -ne 0) { Stop-Install "ComfyUI requirements install failed." }
$managerReq = Join-Path $comfyDir "manager_requirements.txt"
if (Test-Path $managerReq) {
    Write-Host "  Installing ComfyUI-Manager..."
    & $venvPy -m pip install -r $managerReq --quiet
}

# -- 4b. Triton + SageAttention (fast fp8/int8 + attention on Windows) --------
# Triton is the GPU kernel compiler that makes fp8/int8 fast paths (and LoRA on
# quantized checkpoints, e.g. the Depth ControlNet) fast instead of emulated.
# SageAttention (Triton-based INT8/FP8 attention) further speeds up attention.
# triton-windows tracks the PyTorch version: torch 2.12 -> triton 3.7.
& $venvPy -c "import triton,sageattention,sys; sys.exit(0)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing Triton (triton-windows) + SageAttention..."
    & $venvPy -m pip install -U "triton-windows<3.8" --quiet
    & $venvPy -m pip install sageattention --quiet
} else {
    Write-Host "  Triton + SageAttention already present."
}
& $venvPy -c "import triton,sageattention,sys; sys.exit(0)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    WARNING: Triton/SageAttention import failed. Generation still works"
    Write-Host "             (fp8/int8 LoRA will be slower). Re-run install.bat or set KREA_COMFY_SAGE=0."
}

# -- 5. Community Krea-2 custom nodes -----------------------------------------
$nodes = @(
    "ethanfel/ComfyUI-Krea2TextEncoder",
    "nova452/ComfyUI-Conditioning-Rebalance",
    "RamonGuthrie/ComfyUI-RBG-SmartSeedVariance",
    "capitan01R/ComfyUI-Krea2T-Enhancer",
    "Andro-Meta/ComfyUI-Krea-Moodboards",
    "scraed/LanPaint",
    "mozhaa/ComfyUI-Actual-Denoise",
    "Nynxz/ComfyUI-NK2E",
    "AshmoTV/ComfyUi-Untwisting-RoPE-Krea2",
    "blue-pen5805/ComfyUI-krea2-negpip",
    "BobJohnson24/ComfyUI-INT8-Fast",
    "spacepxl/ComfyUI-VAE-Utils",
    "ClownsharkBatwing/RES4LYF",
    "city96/ComfyUI-GGUF",
    "ssitu/ComfyUI_UltimateSDUpscale",
    "numz/ComfyUI-SeedVR2_VideoUpscaler",
    "willmiao/ComfyUI-Lora-Manager",
    "facok/comfyui-krea2-controlnet",
    "Fannovel16/comfyui_controlnet_aux",
    "ltdrdata/ComfyUI-Impact-Pack",
    "ltdrdata/ComfyUI-Impact-Subpack",
    "moonwhaler/comfyui-seedvr2-tilingupscaler",
    "lbouaraba/comfyui-krea2edit",
    "1038lab/ComfyUI-QwenVL",
    "lunaaispace-eng/ComfyUI-DeGrid",
    # kjnodes provides LazySwitchKJ + PathchSageAttentionKJ used by the
    # Turbo 4X workflow template (backend/workflows/turbo_4x_api.json).
    "kijai/ComfyUI-KJNodes"
)
$requiredNodes = @(
    "ComfyUI-Krea2TextEncoder",
    "ComfyUI-Conditioning-Rebalance",
    "ComfyUI-INT8-Fast"
)
$customNodes = Join-Path $comfyDir "custom_nodes"
New-Item -ItemType Directory -Force -Path $customNodes | Out-Null
foreach ($repo in $nodes) {
    $name = $repo.Split("/")[1]
    $dst = Join-Path $customNodes $name
    if (Test-Path $dst) { Write-Host "  custom node present: $name"; continue }
    Write-Host "  cloning custom node $name..."
    & git clone --depth 1 --recurse-submodules "https://github.com/$repo" "$dst"
    if ($LASTEXITCODE -ne 0) {
        if ($requiredNodes -contains $name) { Stop-Install "Required default ComfyUI node failed to clone: $name" }
        Write-Host "    WARNING: clone failed for $name (optional; skipping)."
        continue
    }
    $req = Join-Path $dst "requirements.txt"
    if (Test-Path $req) { & $venvPy -m pip install -r $req --quiet }
}

# -- 5a2. Rebels Mr. Flow (staged-sampling fast upscale) ----------------------
# The node package lives in an inner "ComfyUI-Rebels-MrFlow" subfolder; ComfyUI
# only loads custom_nodes/*/__init__.py one level deep, so relocate it up.
$mrflowPkg = Join-Path $customNodes "ComfyUI-Rebels-MrFlow"
if (-not (Test-Path (Join-Path $mrflowPkg "__init__.py"))) {
    $mrflowClone = Join-Path $customNodes "Rebels_MrFlow"
    if (-not (Test-Path $mrflowClone)) {
        Write-Host "  cloning custom node Rebels_MrFlow..."
        & git clone --depth 1 "https://github.com/RealRebelAI/Rebels_MrFlow.git" "$mrflowClone"
    }
    $inner = Join-Path $mrflowClone "ComfyUI-Rebels-MrFlow"
    if (Test-Path (Join-Path $inner "__init__.py")) {
        if (Test-Path $mrflowPkg) { Remove-Item -Recurse -Force $mrflowPkg }
        Move-Item $inner $mrflowPkg
        Write-Host "    Mr. Flow node package installed."
    } else {
        Write-Host "    WARNING: Rebels_MrFlow layout unexpected (skipping)."
    }
}

# -- 5a3. Mr. Flow SR models (pixel-space upscalers) --------------------------
$upscaleDir = Join-Path $root "models\upscale_models"
New-Item -ItemType Directory -Force -Path $upscaleDir | Out-Null
$srModels = @(
    @{ name = "RealESRGAN_x2plus.pth"; url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" },
    @{ name = "4x_foolhardy_Remacri.pth"; url = "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" }
)
foreach ($m in $srModels) {
    $dest = Join-Path $upscaleDir $m.name
    if (Test-Path $dest) { Write-Host "  SR model present: $($m.name)"; continue }
    Write-Host "  downloading SR model $($m.name)..."
    try { Invoke-WebRequest -Uri $m.url -OutFile $dest -UseBasicParsing }
    catch { Write-Host "    WARNING: download failed for $($m.name)." }
}

# -- 5b. Krea2 wide-range LoRA loader (our own node) --------------------------
# Bypass LoRAs run at extreme weights (up to +/-40000); ComfyUI's stock loader
# caps the widget at +/-100. This tiny node reuses ComfyUI's own LoRA math with
# a wide widget range. It's our code (not a clone), so we write it here.
$wideLoraDir = Join-Path $customNodes "krea2_wide_lora"
New-Item -ItemType Directory -Force -Path $wideLoraDir | Out-Null
$wideLora = @'
"""Wide-range model-only LoRA loader for Krea 2 (bypass LoRAs up to +/-40000)."""
from __future__ import annotations

import folder_paths
import comfy.utils
import comfy.sd


def _normalize_lora_keys(sd: dict) -> dict:
    """Remap the HF PEFT prefix (base_model.model.*) to ComfyUI's diffusion_model.*
    so Krea-2 LoRAs exported by PEFT match the DiT key map instead of silently
    loading 0 layers. Standard keys pass through untouched."""
    if not any(k.startswith("base_model.model.") for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("base_model.model."):
            k = "diffusion_model." + k[len("base_model.model."):]
        out[k] = v
    return out


class Krea2WideLoraLoaderModelOnly:
    _cache = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -40000.0, "max": 40000.0, "step": 0.01,
                    "tooltip": "LoRA weight. Wide range for bypass LoRAs; normal LoRAs use ~-4..4.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Krea2"
    TITLE = "Krea2 Wide LoRA Loader (Model Only)"

    def load(self, model, lora_name, strength_model):
        if strength_model == 0:
            return (model,)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if self.__class__._cache is not None and self.__class__._cache[0] == lora_path:
            lora_sd = self.__class__._cache[1]
        else:
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            lora_sd = _normalize_lora_keys(lora_sd)
            self.__class__._cache = (lora_path, lora_sd)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, strength_model, 0)
        return (model_lora,)


NODE_CLASS_MAPPINGS = {"Krea2WideLoraLoaderModelOnly": Krea2WideLoraLoaderModelOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2WideLoraLoaderModelOnly": "Krea2 Wide LoRA Loader (Model Only)"}
'@
Set-Content -Path (Join-Path $wideLoraDir "__init__.py") -Value $wideLora -Encoding UTF8
Write-Host "  Wrote krea2_wide_lora custom node."

# -- 5c. Krea-2 Real VAE (community realism VAE, HF mirror) -------------------
# Improves fine detail and removes the "textured hair" artifact vs the stock
# Qwen VAE. Used by default when present. SeedVR2's own models auto-download on
# first use via its (Down)Load nodes, so they aren't fetched here.
$realVaeDir = Join-Path $root "models\krea2\vae"
New-Item -ItemType Directory -Force -Path $realVaeDir | Out-Null
$realVae = Join-Path $realVaeDir "krea2RealVae_v10.safetensors"
if (-not (Test-Path $realVae)) {
    Write-Host "  Downloading Krea-2 Real VAE..."
    try {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri "https://huggingface.co/artsyww/KREA2REALVAE/resolve/main/krea2RealVae_v10.safetensors" -OutFile $realVae -TimeoutSec 600
    } catch {
        Write-Host "    WARNING: Krea-2 Real VAE download failed; the stock Qwen VAE will be used instead."
    }
} else {
    Write-Host "  Krea-2 Real VAE already present."
}

# -- 5c2. FaceDetailer detector (God Mode stage 4) ----------------------------
# Impact-Subpack's UltralyticsDetectorProvider reads bbox models from
# ComfyUI/models/ultralytics/bbox (not covered by extra_model_paths).
$bboxDir = Join-Path $comfyDir "models\ultralytics\bbox"
New-Item -ItemType Directory -Force -Path $bboxDir | Out-Null
$faceModel = Join-Path $bboxDir "face_yolov8m.pt"
if (-not (Test-Path $faceModel)) {
    Write-Host "  Downloading FaceDetailer detector (face_yolov8m)..."
    try {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" -OutFile $faceModel -TimeoutSec 300
    } catch { Write-Host "    WARNING: face_yolov8m download failed; God Mode FaceDetailer will be unavailable." }
} else { Write-Host "  FaceDetailer detector already present." }

# -- 5d. Depth ControlNet assets (facok node + DA3 depth model) --------------
# The comfyui-krea2-controlnet node (cloned above) needs the public depth Control
# LoRA (in models/loras) and a Depth-Anything-3 model (in models/geometry_estimation,
# where ComfyUI's core LoadDA3Model reads from).
$ProgressPreference = "SilentlyContinue"
$depthLora = Join-Path $root "models\loras\depth-control-lora.safetensors"
if (-not (Test-Path $depthLora)) {
    Write-Host "  Downloading Krea-2 depth Control LoRA..."
    try {
        Invoke-WebRequest -Uri "https://huggingface.co/Patil/Krea-2-depth-controlnet/resolve/main/depth-control-lora.safetensors" -OutFile $depthLora -TimeoutSec 600
    } catch { Write-Host "    WARNING: depth Control LoRA download failed; Depth Control task will be unavailable until it's fetched." }
} else { Write-Host "  Depth Control LoRA already present." }

$geoDir = Join-Path $root "models\geometry_estimation"
New-Item -ItemType Directory -Force -Path $geoDir | Out-Null
$da3 = Join-Path $geoDir "depth_anything_3_small.safetensors"
if (-not (Test-Path $da3)) {
    Write-Host "  Downloading Depth-Anything-3 (small)..."
    try {
        Invoke-WebRequest -Uri "https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/depth_anything_3_small.safetensors" -OutFile $da3 -TimeoutSec 600
    } catch { Write-Host "    WARNING: Depth-Anything-3 download failed; Depth Control task will be unavailable until it's fetched." }
} else { Write-Host "  Depth-Anything-3 model already present." }

# -- 6. extra_model_paths.yaml (points at <root>/models) ----------------------
$modelsPath = ((Join-Path $root "models") -replace "\\", "/")
$yaml = @"
krea2_backend:
    base_path: $modelsPath

    diffusion_models: |
        krea2/diffusion_models
        pid/checkpoints
        gguf

    # ComfyUI-GGUF derives unet_gguf from diffusion_models and resolves the
    # actual file via the "unet" folder, so gguf must be reachable from both.
    unet: |
        krea2/diffusion_models
        gguf

    text_encoders: krea2/text_encoders

    vae: |
        krea2/vae
        local_ai/qwen_image/vae

    loras: |
        loras
        loras/comfy/v0.1

    checkpoints: pid/checkpoints

    diffusers: local_ai

    unet_gguf: gguf

    geometry_estimation: geometry_estimation

    upscale_models: upscale_models
"@
Set-Content -Path (Join-Path $comfyDir "extra_model_paths.yaml") -Value $yaml -Encoding UTF8
Write-Host "  Wrote extra_model_paths.yaml (models at $modelsPath)."

# -- 7. run_comfyui.bat -------------------------------------------------------
$launcher = @"
@echo off
setlocal
cd /d "%~dp0"
set "KREA_COMFY_ATTENTION_ARGS="
if /I not "%KREA_COMFY_SAGE%"=="0" (
    "%~dp0venv\Scripts\python.exe" -c "import triton,sageattention" >nul 2>&1
    if not errorlevel 1 set "KREA_COMFY_ATTENTION_ARGS=--use-sage-attention"
)
"%~dp0venv\Scripts\python.exe" main.py --enable-manager --port $Port --disable-pinned-memory --reserve-vram 2.0 %KREA_COMFY_ATTENTION_ARGS% %*
pause
"@
Set-Content -Path (Join-Path $comfyDir "run_comfyui.bat") -Value $launcher -Encoding ASCII

Write-Host "  ComfyUI provisioning complete."
exit 0
