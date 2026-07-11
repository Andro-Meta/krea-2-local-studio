# Starts the ComfyUI image engine (if not already running) and waits until it
# answers on /system_stats. Called by run.bat so the whole Krea 2 stack comes
# up with a single command.
param(
    [int]$Port = 8188,
    [int]$TimeoutSec = 180
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$comfyDir = [IO.Path]::GetFullPath((Join-Path $root "ComfyUI"))
$py = [IO.Path]::GetFullPath((Join-Path $comfyDir "venv\Scripts\python.exe"))
$main = [IO.Path]::GetFullPath((Join-Path $comfyDir "main.py"))
. (Join-Path $PSScriptRoot "comfy_process_validation.ps1")

function Get-ComfyHealth {
    param([string]$BaseUrl = "http://127.0.0.1:$Port")
    try {
        $stats = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/system_stats" -TimeoutSec 4
        $versionText = [string]$stats.system.comfyui_version
        $parsedVersion = $null
        if (-not $versionText -or -not [version]::TryParse($versionText, [ref]$parsedVersion)) {
            return $null
        }
        return $stats
    } catch {
        return $null
    }
}

function Get-OwnedComfyListener {
    param([int]$ListenerPid, [int]$StartedPid = 0)
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $ListenerPid" -ErrorAction SilentlyContinue
    $lookup = {
        param($ProcessId)
        Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    }
    $valid = Test-ComfyProcessOwnership -Process $proc -ExpectedPython $py `
        -ExpectedMain $main -StartedPid $StartedPid -ProcessLookup $lookup
    return $(if ($valid) { $proc } else { $null })
}

function Assert-ComfyCancelCapability {
    param([string]$BaseUrl = "http://127.0.0.1:$Port")
    if (-not (Test-ComfyCancelCapability -BaseUrl $BaseUrl)) {
        Write-Host "  ERROR: Atomic job cancellation is unavailable."
        Write-Host "         The installed ComfyUI is too old or mismatched; update/reinstall ComfyUI and retry."
        exit 1
    }
}

function Write-ComfyRuntimeStatus {
    param($Stats, [int]$RuntimePid)
    $argv = @($Stats.system.argv)
    $mode = if ($argv -contains "--highvram") { "HIGH_VRAM" } `
        elseif ($argv -contains "--lowvram") { "LOW_VRAM" } `
        elseif ($argv -contains "--novram") { "NO_VRAM" } `
        else { "NORMAL_VRAM" }
    Write-Host "  Effective ComfyUI: $mode | PID $RuntimePid"
    Write-Host "  Set vram state to: $mode"
}

# Load KREA_COMFY_* tuning flags from .env (run.bat does not export them into
# the environment). Real environment variables keep priority over .env.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*(KREA_COMFY_[A-Z_]+)\s*=\s*(.*?)\s*$') {
            $key = $Matches[1]
            $value = $Matches[2]
            if ($value -and -not [Environment]::GetEnvironmentVariable($key)) {
                [Environment]::SetEnvironmentVariable($key, $value)
            }
        }
    }
}

# KREA_COMFY_URL is the single engine knob: unset/local -> we manage a local
# ComfyUI here; a non-local URL -> the user runs their own engine elsewhere,
# so there is nothing to start on this machine.
$comfyUrl = $env:KREA_COMFY_URL
if ($comfyUrl -and $comfyUrl -notmatch '^https?://(127\.0\.0\.1|localhost)([:/]|$)') {
    Write-Host "  External ComfyUI engine configured at $comfyUrl - not starting a local one."
    if (-not (Get-ComfyHealth -BaseUrl $comfyUrl)) {
        Write-Host "  External ComfyUI is currently unavailable; Krea will retain unavailable-engine behavior."
        exit 0
    }
    if (-not (Test-ComfyCancelCapability -BaseUrl $comfyUrl)) {
        Write-Host "  ERROR: Atomic job cancellation is unavailable on external ComfyUI."
        Write-Host "         The configured ComfyUI is too old or mismatched; update/reinstall it and retry."
        exit 1
    }
    exit 0
}
if ($comfyUrl -match '^https?://(?:127\.0\.0\.1|localhost):(\d+)') {
    $Port = [int]$Matches[1]
}

if (-not (Test-Path $main) -or -not (Test-Path $py)) {
    Write-Host "  ComfyUI not found at $comfyDir - generation cannot start."
    exit 0
}

$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listening) {
    $listenerPid = [int]$listening.OwningProcess
    $stats = Get-ComfyHealth
    if (-not $stats) {
        Write-Host "  ERROR: Port $Port is owned by PID $listenerPid, but /system_stats did not return a valid ComfyUI version."
        Write-Host "         Stop or reconfigure that unrelated process; Krea will not reuse or kill it."
        exit 1
    }
    $confirmedListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $confirmedPid = if ($confirmedListener) { [int]$confirmedListener.OwningProcess } else { 0 }
    if (-not (Test-StableListenerPid -BeforePid $listenerPid -AfterPid $confirmedPid)) {
        Write-Host "  ERROR: Port $Port listener changed during ComfyUI validation ($listenerPid to $confirmedPid)."
        Write-Host "         Krea will not reuse or kill the replacement process."
        exit 1
    }
    $owned = Get-OwnedComfyListener -ListenerPid $listenerPid
    if (-not $owned) {
        Write-Host "  ERROR: PID $listenerPid responds like ComfyUI but is not this repo's managed ComfyUI process."
        Write-Host "         Stop it manually or set KREA_COMFY_URL for an external engine. Krea will not kill an unrelated process."
        exit 1
    }
    Write-Host "  Reusing this repo's ComfyUI listener on port $Port."
    Assert-ComfyCancelCapability
    Write-ComfyRuntimeStatus -Stats $stats -RuntimePid $listenerPid
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
    # VRAM strategy. KREA_COMFY_VRAM_MODE picks ComfyUI's memory mode directly:
    #   highvram   - keep models resident in VRAM (fastest; fine for INT8/fp8 on 24GB)
    #   normalvram - ComfyUI's default balancing (also the default here)
    #   lowvram    - aggressive offload for small cards
    #   novram     - maximum offload (last resort)
    # KREA_COMFY_HIGHVRAM=1 remains as a back-compat alias for highvram.
    $vramMode = "$env:KREA_COMFY_VRAM_MODE".Trim().ToLower()
    if (@("highvram", "normalvram", "lowvram", "novram") -contains $vramMode) {
        if ($vramMode -ne "normalvram") { $extra += "--$vramMode" }
    } elseif ($env:KREA_COMFY_HIGHVRAM -eq "1") {
        $extra += "--highvram"
    }
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
$process = Start-Process -FilePath $py -ArgumentList (@("`"$main`"", "--enable-manager", "--port", "$Port") + $extra) `
    -WorkingDirectory $comfyDir -WindowStyle Hidden `
    -RedirectStandardOutput $comfyOut -RedirectStandardError $comfyErr -PassThru

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    $readyListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $runtimePid = if ($readyListener) { [int]$readyListener.OwningProcess } else { 0 }
    $stats = if ($runtimePid -gt 0) { Get-ComfyHealth } else { $null }
    if ($stats) {
        $confirmedListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        $confirmedPid = if ($confirmedListener) { [int]$confirmedListener.OwningProcess } else { 0 }
        if (-not (Test-StableListenerPid -BeforePid $runtimePid -AfterPid $confirmedPid)) {
            Write-Host "  ERROR: Ready listener changed during ComfyUI validation ($runtimePid to $confirmedPid)."
            Write-Host "         Krea will not kill or reuse the replacement process."
            exit 1
        }
        $owned = if ($runtimePid -gt 0) {
            Get-OwnedComfyListener -ListenerPid $runtimePid -StartedPid $process.Id
        } else {
            $null
        }
        if (-not $owned) {
            Write-Host "  ERROR: Ready listener PID $runtimePid is not the started process or its descendant."
            Write-Host "         Krea will not kill or reuse an unrelated listener."
            exit 1
        }
        Write-Host "  ComfyUI ready."
        Assert-ComfyCancelCapability
        Write-ComfyRuntimeStatus -Stats $stats -RuntimePid $runtimePid
        exit 0
    }
    Start-Sleep -Seconds 2
}
Write-Host "  ERROR: ComfyUI did not return valid /system_stats with a version within $TimeoutSec s."
exit 1
