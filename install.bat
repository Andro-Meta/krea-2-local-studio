@echo off
setlocal EnableDelayedExpansion
title Krea 2 Studio - Install

echo.
echo  ====================================
echo   Krea 2 Studio -- Install
echo  ====================================
echo.

set "ROOT=%~dp0"
cd /d "%ROOT%"

:: -- Python check -------------------------------------------------------------
echo [1/10] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12+ from python.org.
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo        Found Python %PY_VER%
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
    echo ERROR: Python 3.12+ required.
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 12 (
    echo ERROR: Python 3.12+ required, found %PY_VER%.
    exit /b 1
)

:: -- Virtual environment -------------------------------------------------------
echo.
echo [2/10] Creating virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo        venv already exists, skipping.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        exit /b 1
    )
    echo        venv created.
)
call venv\Scripts\activate.bat

:: -- Upgrade pip ---------------------------------------------------------------
echo.
echo [3/10] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: -- PyTorch with CUDA 12.8 ----------------------------------------------------
echo.
echo [4/10] Installing PyTorch + CUDA 12.8 (may take several minutes)...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; assert '+cu' in torch.__version__, torch.__version__; print(torch.__version__)" >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%t in ('python -c "import torch; print(torch.__version__)"') do echo        PyTorch %%t with CUDA already installed, skipping.
) else (
    pip install torch torchvision torchaudio ^
        --index-url https://download.pytorch.org/whl/cu128 ^
        --quiet
    if errorlevel 1 (
        echo WARNING: cu128 wheel failed. Trying cu126 fallback...
        pip install torch torchvision torchaudio ^
            --index-url https://download.pytorch.org/whl/cu126 ^
            --quiet
        if errorlevel 1 (
            echo ERROR: PyTorch installation failed.
            exit /b 1
        )
        echo        Installed PyTorch with CUDA 12.6.
    ) else (
        echo        PyTorch installed with CUDA 12.8.
    )
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('       Verified', torch.__version__, 'on', torch.cuda.get_device_name(0))"
    if errorlevel 1 (
        echo ERROR: PyTorch installed, but CUDA is not available. Update NVIDIA drivers and re-run install.bat.
        exit /b 1
    )
)

:: -- Python dependencies -------------------------------------------------------
echo.
echo [5/10] Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some deps failed. Retrying core deps only...
    pip install fastapi "uvicorn[standard]" python-multipart aiosqlite aiofiles ^
        pydantic pydantic-settings requests pillow numpy einops psutil ^
        transformers safetensors diffusers accelerate torchao websocket-client --quiet
    if errorlevel 1 (
        echo ERROR: Python dependency installation failed.
        exit /b 1
    )
)
echo        Dependencies installed.

:: -- Download Krea support models ------------------------------------------------
echo.
echo [6/10] Downloading Krea support models for moodboards...
echo        This prepares Qwen3-VL conditioning and Qwen-Image VAE assets.
python scripts/download_support_models.py
if errorlevel 1 (
    echo WARNING: Support model download failed.
    echo          Krea can still auto-download these during first model load,
    echo          or use System ^> Krea Moodboard Conditioning to repair.
)

:: -- Download default ComfyUI workflow assets ---------------------------------
echo.
echo [7b/10] Downloading default ComfyUI workflow assets...
echo        This prepares the default Turbo INT8 ConvRot graph, abliterated Krea CLIP,
echo        Qwen VAE fallback, and filter-bypass LoRA used by the default recipe.
python scripts/download_quality_assets.py --assets krea2_turbo_int8_convrot,qwen3vl_abliterated_fp8,qwen_image_comfy_vae,krea2_filter_bypass
if errorlevel 1 (
    echo ERROR: Failed to download one or more default ComfyUI workflow assets.
    echo        Re-run install.bat after checking internet/Hugging Face access.
    exit /b 1
)

:: -- Node.js + Frontend build --------------------------------------------------
echo.
echo [8/10] Building frontend...
node --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Node.js not found. Skipping frontend build.
    echo          Install Node.js 18+ from nodejs.org and re-run install.bat.
    goto :skip_frontend
)
cd frontend
if not exist "node_modules" (
    echo        npm install...
    call npm install --legacy-peer-deps --quiet
    if errorlevel 1 (
        echo ERROR: npm install failed.
        cd ..
        exit /b 1
    )
)
echo        npm run build...
call npm run build
if errorlevel 1 (
    echo ERROR: Frontend build failed.
    cd ..
    exit /b 1
)
cd ..
echo        Frontend built successfully.
goto :done_frontend

:skip_frontend
echo        Frontend skipped.

:done_frontend

:: -- ComfyUI image engine ------------------------------------------------------
echo.
echo [9/10] Setting up ComfyUI image engine (backend)...
echo        Clones ComfyUI, creates its venv, installs PyTorch + requirements,
echo        ComfyUI-Manager, the Krea-2 custom nodes, and extra_model_paths.yaml.
echo        This can take several minutes on first run.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\install_comfyui.ps1"
if errorlevel 1 (
    echo WARNING: ComfyUI provisioning did not complete. You can re-run install.bat,
    echo          run scripts\install_comfyui.ps1 directly, or set KREA_USE_COMFY=0
    echo          in .env to use the in-process native engine instead.
)

:: -- .env scaffold ------------------------------------------------------------
if not exist ".env" (
    echo.
    echo Creating .env from template...
    copy ".env.example" ".env" >nul
    echo        Edit .env -- set HF_TOKEN and model paths.
)

:: -- Tailscale sharing helper -------------------------------------------------
echo.
echo [10/10] Checking Tailscale for public sharing...
where tailscale >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\Tailscale\tailscale.exe" (
        set "TAILSCALE_EXE=C:\Program Files\Tailscale\tailscale.exe"
    ) else (
        echo        Tailscale not found.
        echo        Public sharing uses Tailscale Funnel at /krea.
        choice /c YN /n /t 20 /d N /m "        Install Tailscale with winget now? [Y/N] "
        if errorlevel 2 goto :tailscale_done
        winget install --id Tailscale.Tailscale -e
        goto :tailscale_done
    )
) else (
    set "TAILSCALE_EXE=tailscale"
)
"%TAILSCALE_EXE%" status >nul 2>&1
if errorlevel 1 (
    echo        Tailscale is installed but not connected.
    choice /c YN /n /t 20 /d N /m "        Run tailscale up now? [Y/N] "
    if errorlevel 2 goto :tailscale_done
    "%TAILSCALE_EXE%" up
) else (
    echo        Tailscale is installed and connected.
)
:tailscale_done

:: -- Done ---------------------------------------------------------------------
echo.
echo ====================================
echo  Install complete!
echo.
echo  Next steps:
echo    1. Edit .env -- set HF_TOKEN
echo    2. Run run.bat to start public sharing (also launches ComfyUI)
echo    3. If moodboard conditioning is missing, open System ^> Krea Moodboard Conditioning
echo    4. For local-only mode, run run.bat local
echo    5. ComfyUI is the image engine; set KREA_USE_COMFY=0 in .env for the native engine
echo ====================================
echo.
