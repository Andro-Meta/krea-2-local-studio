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
echo [1/9] Checking Python...
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
echo [2/9] Creating virtual environment...
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
echo [3/9] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: -- PyTorch with CUDA 12.8 ----------------------------------------------------
echo.
echo [4/9] Installing PyTorch + CUDA 12.8 (may take several minutes)...
python -c "import torch; v=torch.__version__; assert '2.' in v" >nul 2>&1
if not errorlevel 1 (
    echo        PyTorch already installed, skipping.
) else (
    pip install torch torchvision torchaudio ^
        --index-url https://download.pytorch.org/whl/cu128 ^
        --quiet
    if errorlevel 1 (
        echo WARNING: cu128 wheel failed. Trying cu121 fallback...
        pip install torch torchvision torchaudio ^
            --index-url https://download.pytorch.org/whl/cu121 ^
            --quiet
        if errorlevel 1 (
            echo ERROR: PyTorch installation failed.
            exit /b 1
        )
        echo        Installed PyTorch with CUDA 12.1.
    ) else (
        echo        PyTorch installed with CUDA 12.8.
    )
)

:: -- Python dependencies -------------------------------------------------------
echo.
echo [5/9] Installing Python dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some deps failed. Retrying core deps only...
    pip install fastapi "uvicorn[standard]" python-multipart aiosqlite aiofiles ^
        pydantic pydantic-settings requests pillow einops ^
        transformers safetensors diffusers accelerate torchao --quiet
)
echo        Dependencies installed.

:: -- Download Krea support models ------------------------------------------------
echo.
echo [6/9] Downloading Krea support models for moodboards...
echo        This prepares Qwen3-VL conditioning and Qwen-Image VAE assets.
python scripts/download_support_models.py
if errorlevel 1 (
    echo WARNING: Support model download failed.
    echo          Krea can still auto-download these during first model load,
    echo          or use System ^> Krea Moodboard Conditioning to repair.
)

:: -- Download Krea 2 source ----------------------------------------------------
echo.
echo [7/9] Downloading Krea 2 model source files...
python scripts/download_krea2.py
if errorlevel 1 (
    echo ERROR: Failed to download krea2 source files.
    exit /b 1
)

:: -- Node.js + Frontend build --------------------------------------------------
echo.
echo [8/9] Building frontend...
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

:: -- .env scaffold ------------------------------------------------------------
if not exist ".env" (
    echo.
    echo Creating .env from template...
    copy ".env.example" ".env" >nul
    echo        Edit .env -- set HF_TOKEN and model paths.
)

:: -- Tailscale sharing helper -------------------------------------------------
echo.
echo [9/9] Checking Tailscale for public sharing...
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
echo    2. Run run.bat to start public sharing
echo    3. If moodboard conditioning is missing, open System ^> Krea Moodboard Conditioning
echo    4. For local-only mode, run run.bat local
echo ====================================
echo.
