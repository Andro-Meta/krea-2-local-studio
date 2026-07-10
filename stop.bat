@echo off
setlocal
title Krea 2 Studio - Stop
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "KREA_PYTHON=venv\Scripts\python.exe"

echo Stopping Krea 2 Studio (backend + ComfyUI + sharing)...
if exist "%KREA_PYTHON%" (
    %KREA_PYTHON% scripts\startup_cleanup.py --wait-seconds 10
) else (
    echo ERROR: venv not found.
)
echo Done. GPU VRAM and system RAM should now be freed.
timeout /t 3 >nul
