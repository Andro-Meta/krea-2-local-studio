# Starts the ComfyUI image engine (if not already running) and waits until it
# answers on /system_stats. Called by run.bat so the whole Krea 2 stack comes
# up with a single command. No-op if KREA_USE_COMFY=0 or ComfyUI isn't present.
param(
    [int]$Port = 8188,
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$comfyDir = Join-Path $root "ComfyUI"
$py = Join-Path $comfyDir "venv\Scripts\python.exe"
$main = Join-Path $comfyDir "main.py"

if (-not (Test-Path $main) -or -not (Test-Path $py)) {
    Write-Host "  ComfyUI not found at $comfyDir - skipping (set KREA_USE_COMFY=0 to use the native engine)."
    exit 0
}

$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    Write-Host "  ComfyUI already running on port $Port."
    exit 0
}

Write-Host "  Starting ComfyUI image engine on port $Port..."
$env:KREA_COMFY_URL = "http://127.0.0.1:$Port"
# RAM tuning:
#  --disable-pinned-memory frees ~13GB of system RAM ComfyUI reserves for
#    weight offloading (a 24GB GPU running Krea turbo doesn't need it).
#  --highvram keeps the model resident in VRAM (no CPU offload churn) - faster
#    and less RAM use. Fine for turbo/int8/fp8; set KREA_COMFY_HIGHVRAM=0 if you
#    run RAW bf16 and hit VRAM OOM. KREA_COMFY_ARGS fully overrides these.
if ($env:KREA_COMFY_ARGS) {
    $extra = $env:KREA_COMFY_ARGS -split '\s+'
} else {
    $extra = @("--disable-pinned-memory")
    # --highvram is OFF by default now: on a 24GB card it pins the DiT + the 5GB
    # Qwen3-VL text encoder + VAE resident, leaving no headroom for activations so
    # the driver spills to shared RAM (catastrophic with fp8 + stacked LoRAs / the
    # depth ControlNet). Default lets ComfyUI offload the text encoder during
    # sampling. Set KREA_COMFY_HIGHVRAM=1 to force it back on.
    if ($env:KREA_COMFY_HIGHVRAM -eq "1") { $extra += "--highvram" }
    # --reserve-vram keeps a headroom buffer so ComfyUI never allocates into the
    # zone where the NVIDIA driver spills VRAM into shared system RAM (the cause
    # of catastrophic minutes-per-step slowdowns with fp8 + stacked LoRAs, e.g.
    # the depth ControlNet). Override with KREA_COMFY_RESERVE_VRAM.
    $reserve = if ($env:KREA_COMFY_RESERVE_VRAM) { $env:KREA_COMFY_RESERVE_VRAM } else { "2.0" }
    $extra += @("--reserve-vram", $reserve)
    # SageAttention (Triton-based INT8/FP8 attention) speeds up the attention
    # layers. Requires triton + sageattention in the ComfyUI venv (installed by
    # install.bat). Set KREA_COMFY_SAGE=0 to disable if it ever misbehaves.
    if ($env:KREA_COMFY_SAGE -ne "0") {
        & $py -c "import triton,sageattention" *> $null
        if ($LASTEXITCODE -eq 0) {
            $extra += "--use-sage-attention"
        } else {
            Write-Host "  SageAttention/Triton not importable in ComfyUI venv - starting with SDPA attention."
        }
    }
}
# Capture ComfyUI's stdout/stderr to log files so the full engine log (node
# loads, generation, tracebacks) is persisted and can be mirrored into the main
# run.bat terminal by run_with_log --tail. Without this, ComfyUI's output only
# lived in a throwaway minimized window.
$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
$comfyOut = Join-Path $logDir "comfyui.log"
$comfyErr = Join-Path $logDir "comfyui.err.log"
# Hidden window: ComfyUI's output is already captured to the log files below
# and mirrored into the main run.bat console by run_with_log --tail, so its
# own console window adds nothing. Keeps run.bat to a single visible terminal.
Start-Process -FilePath $py -ArgumentList (@("main.py", "--enable-manager", "--port", "$Port") + $extra) `
    -WorkingDirectory $comfyDir -WindowStyle Hidden `
    -RedirectStandardOutput $comfyOut -RedirectStandardError $comfyErr | Out-Null

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/system_stats" -UseBasicParsing -TimeoutSec 2 | Out-Null
        Write-Host "  ComfyUI ready."
        exit 0
    } catch {
        Start-Sleep -Seconds 2
    }
}
Write-Host "  WARNING: ComfyUI did not respond within $TimeoutSec s. It may still be loading; generation will work once it finishes."
exit 0
