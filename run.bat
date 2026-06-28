@echo off
setlocal EnableDelayedExpansion
title Krea 2 Studio

set "ROOT=%~dp0"
cd /d "%ROOT%"

if /I "%~1"=="local" goto :local

:: -- Default: login-gated web app with sharing controls ------------------------
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Run install.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

if not exist "backend\krea2\mmdit.py" (
    echo Downloading krea2 source files...
    python scripts\download_krea2.py
    if errorlevel 1 (
        echo ERROR: Could not download krea2/mmdit.py. Check internet connection.
        pause
        exit /b 1
    )
)

echo Stopping any old Krea sharing/server process...
python scripts\startup_cleanup.py --wait-seconds 20
if errorlevel 1 (
    echo WARNING: Some old Krea processes could not be stopped. Startup may still fail or memory may remain in use.
)

for /f "usebackq tokens=*" %%a in (`python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"`) do set "KREA_SERVER_PORT=%%a"

python scripts\download_support_models.py --check >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Local AI/moodboard assets are missing.
    echo          Open System ^> Krea Moodboard Conditioning / Local AI Assets
    echo          or run: venv\Scripts\python.exe scripts\download_support_models.py
    echo.
)

set "KREA_PUBLIC_BASE_PATH=/krea"
set "KREA_SHARE_AUTH_FILE=%ROOT%share_auth.json"
set "KREA_SHARE_SECRET=%RANDOM%%RANDOM%%RANDOM%%RANDOM%%RANDOM%"
set "KREA_SHARE_AUTH_CONFIG="
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        if /I "%%a"=="KREA_SHARE_AUTH" set "KREA_SHARE_AUTH_CONFIG=%%b"
    )
)
for /f "usebackq tokens=*" %%a in (`python -c "import sys; from pathlib import Path; sys.path.insert(0,'backend'); import share_auth; p=Path(r'%ROOT%share_auth.json'); cfg=r'%KREA_SHARE_AUTH_CONFIG%' or None; print('1' if share_auth.resolve_auth_enabled(cfg, has_users=bool(share_auth.load_users(p))) else '0')"`) do set "KREA_SHARE_AUTH=%%a"
if "%KREA_SHARE_AUTH%"=="1" (
    for /f "usebackq tokens=*" %%a in (`python -c "import secrets,sys; from pathlib import Path; sys.path.insert(0,'backend'); import share_auth; p=Path(r'%ROOT%share_auth.json'); users=share_auth.load_users(p); print('') if users else (lambda pw: (share_auth.add_user(p,'admin',pw,role='admin'), print('FIRST_ADMIN_PASSWORD='+pw)))(secrets.token_urlsafe(10))"`) do set "BOOTSTRAP_LOGIN=%%a"
)

echo Starting Krea 2 Studio web sharing mode...
echo.
echo Admin sharing controls are in System ^> Tailscale Sharing.
echo Public Funnel path is always /krea.
if "%KREA_SHARE_AUTH%"=="0" echo Login gate is off because no users are configured.
if defined BOOTSTRAP_LOGIN echo %BOOTSTRAP_LOGIN%
echo.
echo For local-only mode, run:
echo   run.bat local
echo.
set "KREA_SHARE_AUTO_FUNNEL=false"
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        if /I "%%a"=="KREA_SHARE_AUTO_FUNNEL" set "KREA_SHARE_AUTO_FUNNEL=%%b"
    )
)
set "KREA_SHARE_STARTUP_ARGS=--ready-url http://127.0.0.1:%KREA_SERVER_PORT%/krea/api/auth/me --open-url http://localhost:%KREA_SERVER_PORT%/krea --timeout 180"
for /f "usebackq tokens=*" %%a in (`python -c "import sys; from pathlib import Path; sys.path.insert(0,'backend'); import share_auth; p=Path(r'%KREA_SHARE_AUTH_FILE%'); print('1' if share_auth.resolve_auto_funnel_enabled(r'%KREA_SHARE_AUTO_FUNNEL%', auth_enabled=('%KREA_SHARE_AUTH%'=='1'), has_admin=share_auth.has_admin(p)) else '0')"`) do set "KREA_SHARE_AUTO_FUNNEL_ENABLED=%%a"
if "%KREA_SHARE_AUTO_FUNNEL_ENABLED%"=="1" set "KREA_SHARE_STARTUP_ARGS=%KREA_SHARE_STARTUP_ARGS% --auto-funnel"
if not "%KREA_SHARE_AUTO_FUNNEL_ENABLED%"=="1" if /I not "%KREA_SHARE_AUTO_FUNNEL%"=="false" echo Public Funnel auto-start is off because login gate is off or no admin exists.
start "" /b python scripts\share_startup.py %KREA_SHARE_STARTUP_ARGS%
echo Local sharing server: http://localhost:%KREA_SERVER_PORT%/krea
if not exist "logs" mkdir logs
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"`) do set "KREA_LOG_STAMP=%%a"
set "KREA_SERVER_LOG=logs\server-%KREA_LOG_STAMP%.log"
echo Server log: %KREA_SERVER_LOG%
echo ==== Krea server start %DATE% %TIME% ==== > "%KREA_SERVER_LOG%"
python -c "import sys,platform; print('python_executable='+sys.executable); print('python_version='+platform.python_version()); import torch; print('torch='+torch.__version__); print('cuda='+str(torch.cuda.is_available())); print('gpu='+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))" >> "%KREA_SERVER_LOG%" 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:PYTHONUNBUFFERED='1'; python -u -m uvicorn backend.main:app --host 127.0.0.1 --port %KREA_SERVER_PORT% --log-level info 2>&1 | Tee-Object -FilePath '%KREA_SERVER_LOG%' -Append"
exit /b %ERRORLEVEL%

:local
:: -- Preflight: venv -----------------------------------------------------------
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Run install.bat first.
    exit /b 1
)
call venv\Scripts\activate.bat

:: -- Preflight: mmdit.py -------------------------------------------------------
if not exist "backend\krea2\mmdit.py" (
    echo Downloading krea2 source files...
    python scripts\download_krea2.py
    if errorlevel 1 (
        echo ERROR: Could not download krea2/mmdit.py. Check internet connection.
        exit /b 1
    )
)

echo Stopping any old Krea sharing/server process...
python scripts\startup_cleanup.py --wait-seconds 20
if errorlevel 1 (
    echo WARNING: Some old Krea processes could not be stopped. Startup may still fail or memory may remain in use.
)

:: -- Preflight: Krea support models --------------------------------------------
python scripts\download_support_models.py --check >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Krea moodboard conditioning assets are missing.
    echo          Run this to download/repair them:
    echo            venv\Scripts\python.exe scripts\download_support_models.py
    echo          Or open System ^> Krea Moodboard Conditioning in the GUI.
    echo          First model load may also auto-download them.
    echo.
)

:: -- Port conflict check -------------------------------------------------------
echo Checking port 8200...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\port_owner.ps1" -Port 8200 >nul 2>&1
if not errorlevel 1 (
    echo Port 8200 is free.
) else (
    echo.
    echo WARNING: Port 8200 may be in use:
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\port_owner.ps1" -Port 8200
    echo.
    echo Press any key to start anyway, or Ctrl+C to cancel.
    pause >nul
)

:: -- Detect network addresses -------------------------------------------------
for /f "usebackq tokens=*" %%a in (
    `powershell -NoProfile -Command "try { (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object -First 1).IPAddress } catch { '' }"`
) do set LAN_IP=%%a

for /f "usebackq tokens=*" %%a in (
    `powershell -NoProfile -Command "try { (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '100.*' } | Select-Object -First 1).IPAddress } catch { '' }"`
) do set TAILSCALE_IP=%%a

:: -- Firewall rule (idempotent) ------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "if (-not (Get-NetFirewallRule -DisplayName 'Krea2Studio' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'Krea2Studio' -Direction Inbound -Protocol TCP -LocalPort 8200 -Action Allow | Out-Null; Write-Host '  Firewall rule added.' }" 2>nul

:: -- Print access URLs --------------------------------------------------------
echo.
echo ====================================
echo  Krea 2 Studio
echo.
echo    Local:      http://localhost:8200
if defined LAN_IP (
    echo    LAN:        http://%LAN_IP%:8200
)
if defined TAILSCALE_IP (
    echo    Tailscale:  http://%TAILSCALE_IP%:8200
) else (
    echo    Tailscale:  (not connected -- run tailscale up)
)
echo.
echo  Public sharing:
echo    Default run.bat opens the sharing UI. Public Funnel auto-start requires login to be enabled.
echo    This local-only mode is available as: run.bat local
echo.
echo  Press Ctrl+C to stop the server.
echo ====================================
echo.

:: -- Open browser when the server is ready ------------------------------------
start "" /b python scripts\share_startup.py --ready-url http://127.0.0.1:8200/api/system --open-url http://localhost:8200 --timeout 120

:: -- Start server -------------------------------------------------------------
if not exist "logs" mkdir logs
for /f "usebackq tokens=*" %%a in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"`) do set "KREA_LOG_STAMP=%%a"
set "KREA_SERVER_LOG=logs\server-local-%KREA_LOG_STAMP%.log"
echo Server log: %KREA_SERVER_LOG%
echo ==== Krea local server start %DATE% %TIME% ==== > "%KREA_SERVER_LOG%"
python -c "import sys,platform; print('python_executable='+sys.executable); print('python_version='+platform.python_version()); import torch; print('torch='+torch.__version__); print('cuda='+str(torch.cuda.is_available())); print('gpu='+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))" >> "%KREA_SERVER_LOG%" 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:PYTHONUNBUFFERED='1'; python -u -m uvicorn backend.main:app --host 0.0.0.0 --port 8200 --log-level info 2>&1 | Tee-Object -FilePath '%KREA_SERVER_LOG%' -Append"

echo.
echo Server stopped.
